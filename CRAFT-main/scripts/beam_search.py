import os
import operator
import itertools
import re
import json
from tqdm.auto import tqdm

import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem

import torch
import torchtext.vocab.vocab as Vocab
import torch.nn.functional as F

from chemreact.data.rdkit_ops import smiles_to_graph
from chemreact.models.chem_g2s import ChemG2S

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('/media/aita4090/YRF/CRAFT-main/data/label_template.json') as f:
    r_dict = json.load(f)

# -------------------- SMARTS 左侧解析与严格匹配 --------------------
def _is_outer_wrapped_by_parens(s: str) -> bool:
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return False
    pdepth = 0
    bdepth = 0
    for i, ch in enumerate(s):
        if ch == '[':
            bdepth += 1
        elif ch == ']':
            bdepth = max(0, bdepth - 1)
        elif bdepth == 0:
            if ch == '(':
                pdepth += 1
            elif ch == ')':
                pdepth -= 1
                if pdepth == 0 and i != len(s) - 1:
                    return False
    return pdepth == 0

def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while _is_outer_wrapped_by_parens(s):
        s = s[1:-1].strip()
    return s

def _split_components_top_level(left: str):
    parts, cur = [], []
    pdepth = 0
    bdepth = 0
    for ch in left:
        if ch == '[':
            bdepth += 1; cur.append(ch); continue
        if ch == ']':
            bdepth = max(0, bdepth - 1); cur.append(ch); continue
        if bdepth == 0:
            if ch == '(':
                pdepth += 1; cur.append(ch); continue
            if ch == ')':
                pdepth -= 1; cur.append(ch); continue
            if ch == '.' and pdepth == 0:
                seg = "".join(cur).strip()
                if seg:
                    parts.append(seg)
                cur = []
                continue
        cur.append(ch)
    seg = "".join(cur).strip()
    if seg:
        parts.append(seg)
    return parts

def _get_reactant_queries(smarts: str):
    s = smarts.strip()
    if ">>" in s:
        rxn = AllChem.ReactionFromSmarts(s, useSmiles=False)
        if rxn is None:
            return []
        return list(rxn.GetReactants())
    left = _strip_outer_parens(s)
    comps = _split_components_top_level(left)
    qs = []
    for c in comps:
        q = Chem.MolFromSmarts(_strip_outer_parens(c))
        if q is not None:
            qs.append(q)
    return qs

def _mapped_indices_from_match(q: rdchem.Mol, mt):
    out = {}
    for q_idx, tgt_idx in enumerate(mt):
        m = q.GetAtomWithIdx(q_idx).GetAtomMapNum()
        if m:
            out[int(m)] = int(tgt_idx)
    return out

def _strict_select(mol, queries):
    for ci, q in enumerate(queries, start=1):
        mt = mol.GetSubstructMatch(q)
        if mt:
            md = _mapped_indices_from_match(q, mt)
            parts = sorted(md.values())
            return ci, md, parts
    return None

# -------------------- 原子 token 判定与 token→atom 映射 --------------------
_ATOM_MULTI = {"Br", "Cl"}
_ATOM_UP = {"B", "C", "N", "O", "P", "S", "F", "I"}
_ATOM_LOW = {"c", "n", "o", "s"}

def _is_atom_token(tok: str) -> bool:
    if not tok: return False
    if tok.startswith("[") and tok.endswith("]"): return True
    if tok in _ATOM_MULTI: return True
    if tok in _ATOM_UP: return True
    if tok in _ATOM_LOW: return True
    return False

def _build_token2atom(tokens, mol: Chem.Mol) -> torch.Tensor:
    L = len(tokens)
    n_atoms = mol.GetNumAtoms()
    arr = torch.full((L,), -1, dtype=torch.long)
    atom_pos = [i for i, t in enumerate(tokens) if _is_atom_token(t)]
    K = min(len(atom_pos), n_atoms)
    for k in range(K):
        arr[atom_pos[k]] = k
    return arr

# -------------------- ChemG2S 专用束搜索 --------------------
@torch.no_grad()
def _chem_beam_decode_core(model, memory, mem_key_pad, bos_id, eos_id, pad_id,
                           beam_width=10, nbest=5, max_len=256, temperature=1.0, device=None):
    class _Node:
        __slots__=("seq","logp","L")
        def __init__(self, seq, logp, L): self.seq=seq; self.logp=logp; self.L=L
        def score(self, a=0.6): return self.logp/(((5+self.L)/6)**a)

    beams=[_Node([bos_id],0.0,0)]
    finished=[]
    for _ in range(max_len):
        if not beams: break
        B=len(beams)
        tgt_in = torch.tensor([b.seq for b in beams], dtype=torch.long, device=device)  # [B, t]
        mem_rep = memory.repeat(B,1,1)
        mask_rep = mem_key_pad.repeat(B,1)
        logits = model.decoder(tgt_in, mem_rep, memory_key_padding_mask=mask_rep)  # [B,t,V]
        last = logits[:, -1, :] / max(1e-6, temperature)
        # 禁止 <pad>, <bos>, <unk>
        smiles_vocab = getattr(model, "_chem_resources", {}).get('smiles_vocab', {})
        unk_id = smiles_vocab.get('<unk>')
        if unk_id is not None:
            last[:, unk_id] = -1e9
        last[:, [pad_id, bos_id]] = -1e9
        logp = torch.log_softmax(last, dim=-1)

        new=[]
        for i,b in enumerate(beams):
            topk_logp, topk_idx = torch.topk(logp[i], k=min(beam_width, logp.size(1)))
            for k in range(topk_idx.size(0)):
                tok = int(topk_idx[k].item()); lp = float(topk_logp[k].item())
                seq = b.seq + [tok]
                nd = _Node(seq, b.logp+lp, b.L+1)
                if tok==eos_id: finished.append(nd)
                else: new.append(nd)

        if not new and finished: break
        new.sort(key=lambda n: -n.score())
        beams = new[:beam_width]
        if not beams and finished: break

    if len(finished)<nbest: finished.extend(beams)
    finished.sort(key=lambda n: -n.score())

    outs=[]
    for n in finished[:nbest]:
        seq = n.seq
        if seq and seq[0]==bos_id: seq=seq[1:]
        if seq and seq[-1]==eos_id: seq=seq[:-1]
        outs.append(seq)
    return outs

def beam_decode(v: Vocab, model=None, input_tokens=None, template_idx=None,
                device=None, inf_max_len=None, beam_width=10, nbest=5, Temp=None,
                beam_templates:list=None):

    is_chem = isinstance(model, ChemG2S) or (hasattr(model, 'encoder') and hasattr(model, 'role_mgr') and hasattr(model, 'tpl_emb'))
    if not is_chem:
        # ===== baseline 原实现（保持不变）=====
        SOS_token = v['<bos>']; EOS_token = v['<eos>']
        if template_idx is not None:
            template_idx = re.sub(r'\D', '', template_idx)
            if beam_templates is not None and template_idx not in beam_templates:
                beam_width = 5; nbest = 1
        encoder_input = input_tokens
        with torch.no_grad():
            encoder_input = encoder_input.unsqueeze(-1)
            encoder_output, memory_pad_mask = model.encode(encoder_input, src_pad_mask=True)
        decoder_input = torch.tensor([[SOS_token]])
        counter = itertools.count()
        node = type("BeamSearchNode",(object,),{})()
        node.prevNode=None; node.dec_in=decoder_input; node.logp=0; node.leng=0
        with torch.no_grad():
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(decoder_input.size(1)).to(device)
            logits = model.decode(memory=encoder_output, tgt=decoder_input.permute(1, 0).to(device),
                                  tgt_mask=tgt_mask, memory_pad_mask=memory_pad_mask)
            logits = logits.permute(1, 0, 2)
            decoder_output = torch.log_softmax(logits[:, -1, :]/(1.0 if Temp is None else Temp), dim=1).to('cpu')
        tmp_beam_width = min(beam_width, decoder_output.size(1))
        log_prob, indices = torch.topk(decoder_output, tmp_beam_width)
        nextnodes = []
        for new_k in range(tmp_beam_width):
            decoded_t = indices[0][new_k].view(1, -1)
            log_p = log_prob[0][new_k].item()
            next_decoder_input = torch.cat([node.dec_in, decoded_t],dim=1)
            nn_ = type("BeamSearchNode",(object,),{})()
            nn_.prevNode=node; nn_.dec_in=next_decoder_input; nn_.logp=node.logp + log_p; nn_.leng=node.leng + 1
            score = -(nn_.logp/(((5+nn_.leng)/(5+1))**0.6)); count = next(counter)
            nextnodes.append((score, count, nn_))
        for i in range((inf_max_len or 256) - 1):
            if i == 0:
                current_nodes = sorted(nextnodes)[:tmp_beam_width]
            else:
                current_nodes = sorted(nextnodes)[:beam_width]
            nextnodes=[]; scores=[]; counts=[]; nodes=[]; decoder_inputs=[]
            for score, count, node in current_nodes:
                if node.dec_in[0][-1].item() == EOS_token:
                    nextnodes.append((score, count, node))
                else:
                    scores.append(score); counts.append(count); nodes.append(node); decoder_inputs.append(node.dec_in)
            if not bool(decoder_inputs): break
            decoder_inputs = torch.vstack(decoder_inputs)
            enc_out = encoder_output.repeat(1, decoder_inputs.size(0), 1)
            mask = memory_pad_mask.repeat(decoder_inputs.size(0), 1)
            with torch.no_grad():
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(decoder_inputs.size(1)).to(device)
                logits = model.decode(memory=enc_out, tgt=decoder_inputs.permute(1, 0).to(device),
                                      tgt_mask=tgt_mask, memory_pad_mask=mask)
                logits = logits.permute(1, 0, 2)
                decoder_output = torch.log_softmax(logits[:, -1, :]/(1.0 if Temp is None else Temp), dim=1).to('cpu')
            for beam, score in enumerate(scores):
                for token in range(EOS_token, decoder_output.size(-1)):
                    decoded_t = torch.tensor([[token]])
                    log_p = decoder_output[beam, token].item()
                    next_decoder_input = torch.cat([nodes[beam].dec_in, decoded_t],dim=1)
                    nd = type("BeamSearchNode",(object,),{})()
                    nd.prevNode=nodes[beam]; nd.dec_in=next_decoder_input; nd.logp=nodes[beam].logp + log_p; nd.leng=nodes[beam].leng + 1
                    score = -(nd.logp/(((5+nd.leng)/(5+1))**0.6)); count = next(counter)
                    nextnodes.append((score, count, nd))
        outputs = []
        for score, _, n in sorted(nextnodes, key=operator.itemgetter(0))[:nbest]:
            output = n.dec_in.squeeze(0).tolist()[1:-1]
            output = v.lookup_tokens(output)
            output = ''.join(output)
            outputs.append(output)
        return outputs

    # ===== ChemG2S（混合序列编码器）路径 =====
    res = getattr(model, "_chem_resources", None)
    if res is None:
        raise RuntimeError("ChemG2S 缺少 _chem_resources（smiles_vocab/template_id_vocab/template_roles/label_templates）。")

    smiles_vocab = res['smiles_vocab']                 # dict[str->int]
    itos = res['itos']                                 # dict[int->str]
    bos_id = smiles_vocab['<bos>']; eos_id = smiles_vocab['<eos>']; pad_id = smiles_vocab['<pad>']
    tpl_vocab = res['template_id_vocab']               # {tid_str: int}
    tpl_roles = res['template_roles']                  # {tid_str: {"mapnum2role":{...}}}
    label_templates = res['label_templates']           # {tid_str: SMARTS}
    chem_cfg = res.get('chem_cfg', None)

    # baseline tensor → token 序列
    ids = input_tokens.detach().to('cpu').tolist()
    toks = v.lookup_tokens(ids)

    # 模板ID定位与反应物 tokens 裁剪
    tpl_idx = next((i for i, t in enumerate(toks) if re.fullmatch(r"\[\d+\]", t)), None)
    if tpl_idx is None:
        tpl_idx = next((i for i, t in enumerate(toks) if ('[' in t and re.search(r"\d", t))), 0)
    tid_token = toks[tpl_idx]
    tid_str = re.sub(r'\D', '', tid_token) if template_idx is None else re.sub(r'\D', '', template_idx)

    end = len(toks)
    for stop_tok in ('<eos>', '<pad>'):
        try:
            pos = toks.index(stop_tok, tpl_idx + 1)
            end = min(end, pos)
        except ValueError:
            pass
    react_tokens = [t for t in toks[tpl_idx + 1:end] if t not in ('<bos>', '<eos>', '<pad>')]
    react_smi = ''.join(react_tokens)

    if tid_str not in label_templates:
        return []
    mol = Chem.MolFromSmiles(react_smi)
    if mol is None:
        return []
    queries = _get_reactant_queries(label_templates[tid_str])
    sel = _strict_select(mol, queries) if queries else None
    if sel is None:
        return []
    _, mapdict, _ = sel

    # atom_role
    n_atoms = mol.GetNumAtoms()
    map2role = {int(k): int(v) for k, v in tpl_roles[tid_str].get('mapnum2role', {}).items()}
    atom_role = [0] * n_atoms
    for mnum, atom_idx in mapdict.items():
        if 0 <= atom_idx < n_atoms and mnum in map2role:
            atom_role[atom_idx] = map2role[mnum]
    atom_role_t = torch.tensor(atom_role, dtype=torch.long, device=device).unsqueeze(0)  # [1,N]

    # 图与源序列（化学词表）—— 新版：返回 X, edge_index, edge_attr
    atom_feat_dim = chem_cfg["model"]["atom_feat_dim"] if chem_cfg else 32
    X_np, EI_np, EA_np = smiles_to_graph(react_smi, atom_feat_dim)
    X  = torch.from_numpy(X_np).float().unsqueeze(0).to(device)   # [1,N,F]
    EI = torch.from_numpy(EI_np).long().to(device)                # [2,E]
    EA = torch.from_numpy(EA_np).float().to(device)               # [E,edge_dim]
    node_mask = torch.ones(1, n_atoms, dtype=torch.bool, device=device)
    tpl_id_int = torch.tensor([tpl_vocab[tid_str]], dtype=torch.long, device=device)

    # 源 ids（按化学词表逐 token 映射）
    src_ids = torch.tensor([smiles_vocab.get(t, smiles_vocab['<unk>']) for t in react_tokens],
                           dtype=torch.long, device=device).unsqueeze(0)               # [1,Ls]
    src_pad_mask = torch.zeros(1, src_ids.size(1), dtype=torch.bool, device=device)    # [1,Ls]
    token2atom = _build_token2atom(react_tokens, mol).to(device).unsqueeze(0)          # [1,Ls]

    # 角色嵌入
    role_emb = model.role_mgr(tpl_id_int, atom_role_t)  # [1,N,R]

    # 混合序列编码（得到 memory_seq 与 mask）
    memory_seq, mem_pad_seq = model.encoder(
        node_feat=X, atom_role=atom_role_t, role_emb=role_emb, node_mask=node_mask,
        edge_index=EI, edge_attr=EA,
        src_ids=src_ids, src_pad_mask=src_pad_mask, token2atom=token2atom
    )  # [1,Ls,d_model], [1,Ls]

    # 模板 token + memory
    e_tpl = model.tpl_emb(tpl_id_int)          # [1,tpl_dim]
    e_tpl = model.tpl_proj(e_tpl)              # [1,d_model]或Identity
    e_tpl = model.tpl_dropout(e_tpl)
    e_tpl = model.tpl_norm(e_tpl).unsqueeze(1) # [1,1,d_model]
    memory = torch.cat([e_tpl, memory_seq], dim=1)  # [1,1+Ls,d_model]
    mem_pad = torch.cat([torch.zeros(1,1,dtype=torch.bool,device=device), mem_pad_seq], dim=1)

    # 束搜索
    beam_width_eff, nbest_eff = beam_width, nbest
    if template_idx is not None:
        template_idx_num = re.sub(r'\D','', template_idx)
        wl = res.get('beam_templates', None)
        if wl is not None and template_idx_num not in wl:
            beam_width_eff, nbest_eff = 5, 1

    outs = _chem_beam_decode_core(
        model=model, memory=memory, mem_key_pad=mem_pad,
        bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
        beam_width=beam_width_eff, nbest=nbest_eff,
        max_len=(inf_max_len or 256), temperature=(1.0 if Temp is None else Temp),
        device=device
    )

    # id→token→SMILES
    smi_list=[]
    for seq in outs:
        toks = [itos.get(i, '<unk>') for i in seq]
        smi_list.append(''.join(toks))
    return smi_list

def greedy_translate(v: Vocab, model=None, input_tokens=None, device=None, inf_max_len=None):
    is_chem = hasattr(model, 'encoder') and hasattr(model, 'role_mgr') and hasattr(model, 'tpl_emb')
    if not is_chem:
        # baseline 保持不变
        SOS_token = v['<bos>']; EOS_token = v['<eos>']
        encoder_input = input_tokens.permute(1, 0)
        with torch.no_grad():
            enc_out, memory_pad_mask = model.encode(encoder_input, src_pad_mask=True)
            dec_inp = torch.tensor([[SOS_token]]).expand(1, encoder_input.size(1)).to(device)
            EOS_dic = {i: False for i in range(encoder_input.size(1))}
            for _ in range((inf_max_len or 256) - 1):
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(dec_inp.size(0)).to(device)
                logits = model.decode(memory=enc_out, tgt=dec_inp, tgt_mask=tgt_mask, memory_pad_mask=memory_pad_mask)
                dec_out = torch.softmax(logits[-1, :, :], dim=1)
                next_items = dec_out.topk(1)[1].permute(1, 0)
                EOS_indices = (next_items == EOS_token)
                for j, EOS in enumerate(EOS_indices[0]):
                    if EOS: EOS_dic[j] = True
                dec_inp = torch.cat([dec_inp, next_items], dim=0)
                if sum(EOS_dic.values()) == encoder_input.size(1): break
            out = dec_inp.permute(1, 0).to('cpu')
            outputs = []
            for i in range(out.size(0)):
                out_tokens = v.lookup_tokens(out[i].tolist())
                try:
                    eos_idx = out_tokens.index('<eos>')
                    out_tokens = out_tokens[1:eos_idx]
                    outputs.append(''.join(out_tokens))
                except ValueError:
                    continue
        return outputs

    # —— ChemG2S 路径（与 beam_decode 的memory构造一致，解码贪心）——
    res = getattr(model, "_chem_resources", None)
    if res is None:
        raise RuntimeError("ChemG2S 缺少 _chem_resources。")

    smiles_vocab = res['smiles_vocab']; itos = res['itos']
    bos_id = smiles_vocab['<bos>']; eos_id = smiles_vocab['<eos>']; pad_id = smiles_vocab['<pad>']
    tpl_vocab = res['template_id_vocab']; tpl_roles = res['template_roles']; label_templates = res['label_templates']
    chem_cfg = res.get('chem_cfg', None)

    # 支持批/单样本
    if input_tokens.dim() == 1:
        seq_list = [input_tokens]
    elif input_tokens.dim() == 2:
        if input_tokens.size(0) <= input_tokens.size(1):
            seq_list = [input_tokens[i, :] for i in range(input_tokens.size(0))]
        else:
            seq_list = [input_tokens[:, i] for i in range(input_tokens.size(1))]
    else:
        raise ValueError("input_tokens 必须是 1D 或 2D 张量")

    results = []
    for seq_tok in seq_list:
        ids = seq_tok.detach().to('cpu').tolist()
        toks = v.lookup_tokens(ids)
        tpl_pos = next((i for i, t in enumerate(toks) if re.fullmatch(r"\[\d+\]", t)), None)
        if tpl_pos is None:
            tpl_pos = next((i for i, t in enumerate(toks) if ('[' in t and re.search(r"\d", t))), 0)
        tid_token = toks[tpl_pos]; tid_str = re.sub(r'\D', '', tid_token)

        end = len(toks)
        for stop_tok in ('<eos>', '<pad>'):
            try:
                pos = toks.index(stop_tok, tpl_pos + 1); end = min(end, pos)
            except ValueError:
                pass
        react_tokens = [t for t in toks[tpl_pos + 1:end] if t not in ('<bos>', '<eos>', '<pad>')]
        react_smi = ''.join(react_tokens)

        if tid_str not in label_templates: continue
        mol = Chem.MolFromSmiles(react_smi)
        if mol is None: continue
        queries = _get_reactant_queries(label_templates[tid_str])
        sel = _strict_select(mol, queries) if queries else None
        if sel is None: continue
        _, mapdict, _ = sel

        n_atoms = mol.GetNumAtoms()
        map2role = {int(k): int(v) for k, v in tpl_roles[tid_str].get('mapnum2role', {}).items()}
        atom_role = [0] * n_atoms
        for mnum, atom_idx in mapdict.items():
            if 0 <= atom_idx < n_atoms and mnum in map2role:
                atom_role[atom_idx] = map2role[mnum]
        atom_role_t = torch.tensor(atom_role, dtype=torch.long, device=device).unsqueeze(0)

        atom_feat_dim = chem_cfg["model"]["atom_feat_dim"] if chem_cfg else 32
        X_np, EI_np, EA_np = smiles_to_graph(react_smi, atom_feat_dim)
        X  = torch.from_numpy(X_np).float().unsqueeze(0).to(device)
        EI = torch.from_numpy(EI_np).long().to(device)
        EA = torch.from_numpy(EA_np).float().to(device)
        node_mask = torch.ones(1, n_atoms, dtype=torch.bool, device=device)
        tpl_id_int = torch.tensor([tpl_vocab[tid_str]], dtype=torch.long, device=device)

        src_ids = torch.tensor([smiles_vocab.get(t, smiles_vocab['<unk>']) for t in react_tokens],
                               dtype=torch.long, device=device).unsqueeze(0)
        src_pad_mask = torch.zeros(1, src_ids.size(1), dtype=torch.bool, device=device)
        token2atom = _build_token2atom(react_tokens, mol).to(device).unsqueeze(0)

        role_emb = model.role_mgr(tpl_id_int, atom_role_t)
        memory_seq, mem_pad_seq = model.encoder(
            node_feat=X, atom_role=atom_role_t, role_emb=role_emb, node_mask=node_mask,
            edge_index=EI, edge_attr=EA,
            src_ids=src_ids, src_pad_mask=src_pad_mask, token2atom=token2atom
        )
        e_tpl = model.tpl_emb(tpl_id_int)
        e_tpl = model.tpl_proj(e_tpl)
        e_tpl = model.tpl_dropout(e_tpl)
        e_tpl = model.tpl_norm(e_tpl).unsqueeze(1)
        memory = torch.cat([e_tpl, memory_seq], dim=1)
        mem_pad = torch.cat([torch.zeros(1,1,dtype=torch.bool,device=device), mem_pad_seq], dim=1)

        # 贪心解码
        tgt = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        max_len = inf_max_len or 256
        with torch.no_grad():
            for _ in range(max_len):
                logits = model.decoder(tgt, memory, memory_key_padding_mask=mem_pad)
                last = logits[:, -1, :]
                unk_id = smiles_vocab.get('<unk>')
                if unk_id is not None:
                    last[:, unk_id] = -1e9
                last[:, [pad_id, bos_id]] = -1e9
                nxt = int(torch.argmax(last, dim=-1).item())
                tgt = torch.cat([tgt, torch.tensor([[nxt]], device=device)], dim=1)
                if nxt == eos_id:
                    break

        gen_ids = tgt.squeeze(0).tolist()
        if gen_ids and gen_ids[0] == bos_id: gen_ids = gen_ids[1:]
        if gen_ids and gen_ids[-1] == eos_id: gen_ids = gen_ids[:-1]
        tokens = [itos.get(i, '<unk>') for i in gen_ids]
        results.append(''.join(tokens))
    return results
