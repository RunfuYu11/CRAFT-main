import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from tqdm import tqdm

# ===== 项目内模块 =====
from chemreact.utils.config import load_config
from chemreact.data.tokenization import load_vocab, SmilesTokenizer
from chemreact.data.rdkit_ops import smiles_to_graph
from chemreact.models.chem_g2s import ChemG2S

# ======================= 基础IO =======================
def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def parse_test_line(line: str) -> Tuple[str, str, List[str]]:
    parts = line.strip().split()
    import re
    m = re.search(r"\d+", parts[0])
    if not m:
        raise ValueError(f"无法解析模板ID: {line}")
    tid = m.group(0)
    tokens = parts[1:]
    smi = "".join(tokens)
    return tid, smi, tokens

def load_template_vocab_map(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "template_id_to_int" in data:
        m = {str(k): int(v) for k, v in data["template_id_to_int"].items()}
    else:
        m = {str(k): int(v) for k, v in data.items()}
    inv = {v: k for k, v in m.items()}
    return m, inv

# ======================= SMARTS左侧匹配 =======================
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

def _split_components_top_level(left: str) -> List[str]:
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

def get_reactant_queries(template: str) -> List[rdchem.Mol]:
    s = template.strip()
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

def mapped_indices_from_match(q: rdchem.Mol, mt: Tuple[int, ...]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for q_idx, tgt_idx in enumerate(mt):
        m = q.GetAtomWithIdx(q_idx).GetAtomMapNum()
        if m:
            out[int(m)] = int(tgt_idx)
    return out

def select_one_component_by_match(mol: rdchem.Mol, queries: List[rdchem.Mol]) -> Optional[Tuple[str, int, str, Dict[int,int], List[int]]]:
    for ci, q in enumerate(queries, start=1):
        mt = mol.GetSubstructMatch(q)
        if mt:
            md = mapped_indices_from_match(q, mt)
            smi = Chem.MolToSmarts(q, isomericSmiles=True)
            parts = sorted(md.values())
            return ("strict", ci, smi, md, parts)
    return None

# ======================= 原子token判定与映射 =======================
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

def build_token2atom(tokens: List[str], mol: Chem.Mol) -> torch.Tensor:
    L = len(tokens)
    n_atoms = mol.GetNumAtoms()
    arr = torch.full((L,), -1, dtype=torch.long)
    atom_pos = [i for i, t in enumerate(tokens) if _is_atom_token(t)]
    K = min(len(atom_pos), n_atoms)
    for k in range(K):
        arr[atom_pos[k]] = k
    return arr

# ======================= Beam Search =======================
class BeamNode:
    __slots__ = ("seq", "logp", "len")
    def __init__(self, seq: List[int], logp: float, length: int):
        self.seq = seq
        self.logp = float(logp)
        self.len = int(length)
    def score(self, alpha: float = 0.6) -> float:
        return self.logp / (((5 + self.len) / 6) ** alpha)

def logits_mask_forbidden(logits: torch.Tensor, forbid_ids: List[int]):
    if forbid_ids:
        logits[:, forbid_ids] = -1e9
    return logits

@torch.no_grad()
def beam_search_decode(
    model: ChemG2S,
    memory: torch.Tensor,                     # [1,N,d_model]
    mem_key_padding_mask: torch.Tensor,       # [1,N] True=pad
    bos_id: int, eos_id: int, pad_id: int,
    beam_size: int = 5, nbest: int = 1, max_len: int = 128,
    temperature: float = 1.0,
    forbid_ids: Optional[List[int]] = None,
    device: torch.device = torch.device("cpu"),
) -> List[List[int]]:
    forbid_ids = (forbid_ids or [])
    beams: List[BeamNode] = [BeamNode([bos_id], 0.0, 0)]
    finished: List[BeamNode] = []

    for _ in range(max_len):
        tgt_in = torch.tensor([b.seq for b in beams], dtype=torch.long, device=device)  # [B*, t]
        Bcur = tgt_in.size(0)
        mem_rep = memory.repeat(Bcur, 1, 1)
        mask_rep = mem_key_padding_mask.repeat(Bcur, 1)
        logits = model.decoder(tgt_in, mem_rep, memory_key_padding_mask=mask_rep)  # [B*, t, V]
        last = logits[:, -1, :] / max(1e-6, temperature)
        last = logits_mask_forbidden(last, forbid_ids=[pad_id, bos_id])
        log_prob = F.log_softmax(last, dim=-1)

        new_beams: List[BeamNode] = []
        for i, b in enumerate(beams):
            topk_logp, topk_idx = torch.topk(log_prob[i], k=min(beam_size, log_prob.size(1)))
            for k in range(topk_idx.size(0)):
                tok = int(topk_idx[k].item())
                lp = float(topk_logp[k].item())
                seq_new = b.seq + [tok]
                node = BeamNode(seq_new, b.logp + lp, b.len + 1)
                if tok == eos_id:
                    finished.append(node)
                else:
                    new_beams.append(node)

        if not new_beams and finished:
            break
        new_beams.sort(key=lambda x: -x.score())
        beams = new_beams[:beam_size]
        if not beams and finished:
            break

    if len(finished) < nbest:
        finished.extend(beams)
    finished.sort(key=lambda x: -x.score())

    outs: List[List[int]] = []
    for b in finished[:nbest]:
        seq = b.seq
        if seq and seq[0] == bos_id: seq = seq[1:]
        if seq and seq[-1] == eos_id: seq = seq[:-1]
        outs.append(seq)
    return outs

# ======================= 单样本推理（混合序列编码器路径） =======================
@torch.no_grad()
def infer_one(
    cfg: Dict[str, Any],
    model: ChemG2S,
    smiles_vocab: Dict[str, int],
    template_vocab: Dict[str, int],
    label_templates: Dict[str, str],
    test_line: str,
    device: torch.device,
    beam_size: int,
    nbest: int,
    max_len: int,
    temperature: float
) -> Dict[str, Any]:

    itos = {i: s for s, i in smiles_vocab.items()}
    tok = SmilesTokenizer(smiles_vocab)
    pad_id = smiles_vocab["<pad>"]
    bos_id = smiles_vocab["<bos>"]
    eos_id = smiles_vocab["<eos>"]

    # 解析输入
    tpl_id_str, react_smi, react_tokens = parse_test_line(test_line)
    mol = Chem.MolFromSmiles(react_smi)
    if mol is None:
        return {
            "template_id": tpl_id_str, "reactant_tokens": react_tokens, "reactant_smiles": react_smi,
            "match": False, "match_mode": "rdkit_parse_fail", "matched_atoms": [],
            "mapnum_to_atom_idx": {}, "beams_tokens": []
        }
    n_atoms = mol.GetNumAtoms()

    # 模板左侧匹配（strict）
    if tpl_id_str not in label_templates:
        return {
            "template_id": tpl_id_str, "reactant_tokens": react_tokens, "reactant_smiles": react_smi,
            "match": False, "match_mode": "template_not_found", "matched_atoms": [],
            "mapnum_to_atom_idx": {}, "beams_tokens": []
        }
    queries = get_reactant_queries(label_templates[tpl_id_str])
    sel = select_one_component_by_match(mol, queries) if queries else None
    if sel is None:
        return {
            "template_id": tpl_id_str, "reactant_tokens": react_tokens, "reactant_smiles": react_smi,
            "match": False, "match_mode": "fail", "matched_atoms": [],
            "mapnum_to_atom_idx": {}, "beams_tokens": []
        }
    match_mode, comp_idx, smi_used, mapdict, participant = sel

    # atom_role
    with open(cfg["data"]["template_roles"], "r", encoding="utf-8") as f:
        template_roles = json.load(f)
    tr = template_roles[tpl_id_str]
    map2role = {int(k): int(v) for k, v in tr.get("mapnum2role", {}).items()}
    atom_role = [0] * n_atoms
    for mnum, atom_idx in mapdict.items():
        if 0 <= atom_idx < n_atoms and mnum in map2role:
            atom_role[atom_idx] = map2role[mnum]
    atom_role_t = torch.tensor(atom_role, dtype=torch.long, device=device).unsqueeze(0)  # [1,N]

    # 图（新版：返回 X, edge_index, edge_attr）
    X_np, EI_np, EA_np = smiles_to_graph(react_smi, cfg["model"]["atom_feat_dim"])
    X  = torch.from_numpy(X_np).float().unsqueeze(0).to(device)            # [1,N,F]
    EI = torch.from_numpy(EI_np).long().to(device)                         # [2,E]
    EA = torch.from_numpy(EA_np).float().to(device)                        # [E,edge_dim]
    node_mask = torch.ones(1, n_atoms, dtype=torch.bool, device=device)    # [1,N]

    # 源序列 ids 与 token2atom
    src_ids_1d = torch.tensor(tok.encode_tokens(react_tokens, add_bos_eos=False), dtype=torch.long, device=device)  # [Ls]
    token2atom_1d = build_token2atom(react_tokens, mol).to(device)  # [Ls]
    src_ids = src_ids_1d.unsqueeze(0)                 # [1,Ls]
    token2atom = token2atom_1d.unsqueeze(0)           # [1,Ls]
    src_pad_mask = torch.zeros(1, src_ids.size(1), dtype=torch.bool, device=device)  # [1,Ls]

    # 模板ID
    tpl_id_int = torch.tensor([template_vocab[tpl_id_str]], dtype=torch.long, device=device)  # [1]

    # 角色嵌入
    role_emb = model.role_mgr(tpl_id_int, atom_role_t)   # [1,N,R]

    # 编码端（HybridSeqEncoder）
    memory_seq, mem_pad_seq = model.encoder(
        node_feat=X,
        atom_role=atom_role_t,
        role_emb=role_emb,
        node_mask=node_mask,
        edge_index=EI,
        edge_attr=EA,
        src_ids=src_ids,
        src_pad_mask=src_pad_mask,
        token2atom=token2atom
    )  # [1,Ls,d_model], [1,Ls]

    # 模板token + memory拼接（新版：tpl_emb + 可选线性映射 + LN + dropout）
    e_tpl = model.tpl_emb(tpl_id_int)                  # [1,tpl_dim]
    e_tpl = model.tpl_proj(e_tpl)                      # [1,d_model] 或 Identity
    e_tpl = model.tpl_dropout(e_tpl)
    e_tpl = model.tpl_norm(e_tpl).unsqueeze(1)         # [1,1,d_model]
    memory = torch.cat([e_tpl, memory_seq], dim=1)     # [1,1+Ls,d_model]
    mem_key_pad = torch.cat(
        [torch.zeros(1, 1, dtype=torch.bool, device=device), mem_pad_seq],
        dim=1
    )  # [1,1+Ls]

    # 波束搜索
    beams = beam_search_decode(
        model=model,
        memory=memory,
        mem_key_padding_mask=mem_key_pad,
        bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
        beam_size=beam_size, nbest=nbest, max_len=max_len,
        temperature=temperature,
        forbid_ids=[pad_id, bos_id],
        device=device
    )

    beams_tokens = []
    for ids in beams:
        toks = [itos.get(i, "<unk>") for i in ids]
        beams_tokens.append(toks)

    return {
        "template_id": tpl_id_str,
        "reactant_tokens": react_tokens,
        "reactant_smiles": react_smi,
        "match": True,
        "match_mode": match_mode,
        "matched_atoms": participant,
        "mapnum_to_atom_idx": {int(k): int(v) for k, v in mapdict.items()},
        "beams_tokens": beams_tokens
    }

# ======================= CLI =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/media/aita4090/YRF/CRAFT-main/configs/default.yaml")
    ap.add_argument("--checkpoint", default="/media/aita4090/YRF/CRAFT-main/chemreact/train/checkpoints/best.pt")
    ap.add_argument("--label_templates", default="/media/aita4090/YRF/CRAFT-main/data/label_template.json")
    ap.add_argument("--test", default="/media/aita4090/YRF/CRAFT-main/data/input/test.txt")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--nbest", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="/media/aita4090/YRF/CRAFT-main/out/predictions.txt")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # 设备
    dev_name = cfg["misc"]["device"]
    if dev_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_name)

    # 词表与模型
    smiles_vocab = load_vocab(cfg["data"]["smiles_vocab"])
    pad_id = smiles_vocab["<pad>"]
    tpl_vocab, _ = load_template_vocab_map(cfg["data"]["template_id_vocab"])

    model = ChemG2S(cfg, vocab_size=len(smiles_vocab), pad_id=pad_id).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()

    # 模板库
    with open(args.label_templates, "r", encoding="utf-8") as f:
        label_templates = json.load(f)

    # 测试集
    lines = read_lines(args.test)

    # 推理
    outputs: List[str] = []
    for i, line in enumerate(tqdm(lines, ncols=100, desc="inference")):
        try:
            res = infer_one(
                cfg=cfg,
                model=model,
                smiles_vocab=smiles_vocab,
                template_vocab=tpl_vocab,
                label_templates=label_templates,
                test_line=line,
                device=device,
                beam_size=args.beam_size,
                nbest=args.nbest,
                max_len=args.max_len,
                temperature=args.temperature
            )
        except Exception as e:
            outputs.append(f"# line {i+1} 解析失败: {e}")
            continue

        if not res["match"]:
            outputs.append(f"id={res['template_id']} | match=False | reason={res['match_mode']}")
            continue

        head = f"id={res['template_id']} | match=True | mode={res['match_mode']} | atoms={res['matched_atoms']}"
        outputs.append(head)
        for b_idx, toks in enumerate(res["beams_tokens"], start=1):
            outputs.append(f"  beam#{b_idx}: " + " ".join(toks))

    # 写文件
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for ln in outputs:
                f.write(ln + "\n")
    else:
        for ln in outputs:
            print(ln)

if __name__ == "__main__":
    main()
