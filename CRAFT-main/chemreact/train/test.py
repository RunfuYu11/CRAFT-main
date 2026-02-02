import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem

from chemreact.utils.config import load_config, ensure_dirs
from chemreact.utils.logging import human_time
from chemreact.data.tokenization import load_vocab, SmilesTokenizer
from chemreact.data.rdkit_ops import smiles_to_graph
from chemreact.data.dataset import ChemGenDataset
from chemreact.data.collate import collate_fn
from chemreact.models.chem_g2s import ChemG2S


# ======================= 基础IO/解析 =======================
def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def parse_src_line(line: str) -> Tuple[str, List[str]]:
    parts = line.strip().split()
    m = re.search(r"\d+", parts[0])
    if not m:
        raise ValueError(f"无法解析模板ID: {line}")
    tid = m.group(0)
    tokens = parts[1:]
    return tid, tokens

def parse_tgt_smiles_from_tokens(line: str) -> str:
    toks = [t for t in line.strip().split() if t not in ("<bos>", "<eos>", "<pad>")]
    return "".join(toks)

def load_template_vocab_map(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "template_id_to_int" in data:
        return {str(k): int(v) for k, v in data["template_id_to_int"].items()}
    return {str(k): int(v) for k, v in data.items()}

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

def strict_select(mol: rdchem.Mol, queries: List[rdchem.Mol]) -> Optional[Tuple[Dict[int,int], List[int]]]:
    for q in queries:
        mt = mol.GetSubstructMatch(q)
        if mt:
            md = mapped_indices_from_match(q, mt)
            parts = sorted(md.values())
            return md, parts
    return None

# ======================= token→atom 映射 =======================
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
    memory: torch.Tensor,                     # [B,N,d]
    mem_key_padding_mask: torch.Tensor,       # [B,N] True=pad
    bos_id: int, eos_id: int, pad_id: int,
    beam_size: int = 10, nbest: int = 10, max_len: int = 256,
    temperature: float = 1.0,
    forbid_ids: Optional[List[int]] = None,
    device: torch.device = torch.device("cpu"),
) -> List[List[int]]:
    B0 = memory.size(0)
    assert B0 == 1, "仅支持 batch=1 评测"
    forbid_ids = (forbid_ids or [])
    beams: List[BeamNode] = [BeamNode([bos_id], 0.0, 0)]
    finished: List[BeamNode] = []

    for _ in range(max_len):
        tgt_in = torch.tensor([b.seq for b in beams], dtype=torch.long, device=device)  # [B*, t]
        Bcur = tgt_in.size(0)
        mem_rep  = memory.repeat(Bcur, 1, 1)
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

# ======================= 单样本推理 =======================
@torch.no_grad()
def infer_one_beams(
    cfg: Dict[str, Any],
    model: ChemG2S,
    smiles_vocab: Dict[str, int],
    template_vocab: Dict[str, int],
    template_roles: Dict[str, Any],
    label_templates: Dict[str, str],
    src_line: str,
    device: torch.device,
    beam_size: int = 10,
    nbest: int = 10,
    max_len: int = 256,
    temperature: float = 1.0
) -> List[str]:
    itos = {i: s for s, i in smiles_vocab.items()}
    tok = SmilesTokenizer(smiles_vocab)
    pad_id = smiles_vocab["<pad>"]; bos_id = smiles_vocab["<bos>"]; eos_id = smiles_vocab["<eos>"]

    tid_str, react_tokens = parse_src_line(src_line)
    react_smi = "".join([t for t in react_tokens if t not in ("<bos>", "<eos>", "<pad>")])
    if tid_str not in label_templates:
        return []

    mol = Chem.MolFromSmiles(react_smi)
    if mol is None:
        return []

    queries = get_reactant_queries(label_templates[tid_str])
    sel = strict_select(mol, queries) if queries else None
    if sel is None:
        return []
    mapdict, _ = sel

    n_atoms = mol.GetNumAtoms()
    # atom roles
    tr = template_roles.get(tid_str, {})
    map2role = {int(k): int(v) for k, v in tr.get("mapnum2role", {}).items()}
    atom_role = [0] * n_atoms
    for mnum, atom_idx in mapdict.items():
        if 0 <= atom_idx < n_atoms and mnum in map2role:
            atom_role[atom_idx] = map2role[mnum]
    atom_role_t = torch.tensor(atom_role, dtype=torch.long, device=device).unsqueeze(0)  # [1,N]

    # 图（X, edge_index, edge_attr）
    X_np, EI_np, EA_np = smiles_to_graph(react_smi, cfg["model"]["atom_feat_dim"])
    X  = torch.from_numpy(X_np).float().unsqueeze(0).to(device)  # [1,N,F]
    EI = torch.from_numpy(EI_np).long().to(device)               # [2,E]
    EA = torch.from_numpy(EA_np).float().to(device)              # [E,edge_dim]
    node_mask = torch.ones(1, n_atoms, dtype=torch.bool, device=device)

    # 源序列
    src_ids_1d = torch.tensor(tok.encode_tokens(react_tokens, add_bos_eos=False),
                              dtype=torch.long, device=device)        # [Ls]
    token2atom_1d = build_token2atom(react_tokens, mol).to(device)    # [Ls]
    src_ids     = src_ids_1d.unsqueeze(0)                              # [1,Ls]
    token2atom  = token2atom_1d.unsqueeze(0)                           # [1,Ls]
    src_pad_mask= torch.zeros(1, src_ids.size(1), dtype=torch.bool, device=device)

    # 模板ID
    tpl_id_int = torch.tensor([template_vocab[tid_str]], dtype=torch.long, device=device)  # [1]

    # 角色嵌入
    role_emb = model.role_mgr(tpl_id_int, atom_role_t)  # [1,N,R]

    # 编码
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
    )  # [1,Ls,d], [1,Ls]

    # 模板 token
    e_tpl = model.tpl_emb(tpl_id_int)          # [1,tpl_dim]
    e_tpl = model.tpl_proj(e_tpl)              # [1,d]
    e_tpl = model.tpl_dropout(e_tpl)
    e_tpl = model.tpl_norm(e_tpl).unsqueeze(1) # [1,1,d]

    memory = torch.cat([e_tpl, memory_seq], dim=1)  # [1,1+Ls,d]
    mem_key_pad = torch.cat(
        [torch.zeros(1,1,dtype=torch.bool,device=device), mem_pad_seq], dim=1
    )

    # 波束搜索
    seqs = beam_search_decode(
        model=model,
        memory=memory,
        mem_key_padding_mask=mem_key_pad,
        bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
        beam_size=beam_size, nbest=nbest, max_len=max_len,
        temperature=temperature, forbid_ids=[pad_id, bos_id], device=device
    )

    # id→token→SMILES
    out_smis: List[str] = []
    for ids in seqs:
        toks = [itos.get(i, "<unk>") for i in ids]
        out_smis.append("".join(toks))
    return out_smis

# ======================= 分子等价判断 =======================
def canon_smi(s: str) -> Optional[str]:
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=True)

# ======================= 评测（Top-n，解码口径） =======================
def evaluate_topn(
    cfg, model, smiles_vocab, template_vocab, template_roles, label_templates,
    src_lines: List[str], tgt_lines: List[str],
    device: torch.device, beam_size: int = 10, nbest: int = 10, max_len: int = 256, temperature: float = 1.0
):
    assert len(src_lines) == len(tgt_lines)
    N = len(src_lines)
    hit1 = 0
    hit5 = 0
    hit10 = 0
    invalid_gold = 0

    pbar = tqdm(range(N), ncols=120, desc="eval(top-n)")
    for i in pbar:
        gold_smi_raw = parse_tgt_smiles_from_tokens(tgt_lines[i])
        gold_c = canon_smi(gold_smi_raw)
        if gold_c is None:
            invalid_gold += 1
            pbar.set_postfix({"invalid_gold": invalid_gold})
            continue

        preds = infer_one_beams(
            cfg=cfg, model=model, smiles_vocab=smiles_vocab, template_vocab=template_vocab,
            template_roles=template_roles, label_templates=label_templates,
            src_line=src_lines[i], device=device,
            beam_size=beam_size, nbest=nbest, max_len=max_len, temperature=temperature
        )

        preds_c = []
        for s in preds:
            c = canon_smi(s)
            if c is not None:
                preds_c.append(c)

        # Top-1 / 5 / 10
        if len(preds_c) >= 1 and preds_c[0] == gold_c:
            hit1 += 1
        if any(p == gold_c for p in preds_c[:5]):
            hit5 += 1
        if any(p == gold_c for p in preds_c[:10]):
            hit10 += 1

        done = i + 1 - invalid_gold
        if done > 0:
            pbar.set_postfix({
                "top1": f"{hit1/done:.4f}",
                "top5": f"{hit5/done:.4f}",
                "top10": f"{hit10/done:.4f}",
                "valid": done
            })

    denom = N - invalid_gold
    if denom <= 0:
        return 0.0, 0.0, 0.0, invalid_gold
    return hit1/denom, hit5/denom, hit10/denom, invalid_gold

# ======================= CLI =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/media/aita4090/YRF/CRAFT-main/configs/default.yaml")
    ap.add_argument("--checkpoint", default="/media/aita4090/YRF/CRAFT-main/chemreact/train/checkpoints/best.pt")
    ap.add_argument("--src_test", default="/media/aita4090/YRF/CRAFT-main/data/out/src_test.txt")
    ap.add_argument("--tgt_test", default="/media/aita4090/YRF/CRAFT-main/data/out/tgt_test.txt")
    ap.add_argument("--beam_size", type=int, default=10)
    ap.add_argument("--nbest", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true", default=False)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    # 设备
    dev_name = cfg["misc"]["device"]
    device = torch.device(
        "cuda" if (dev_name == "auto" and torch.cuda.is_available())
        else (dev_name if dev_name != "auto" else "cpu")
    )
    print(f"[{human_time()}] device={device}")

    # 词表与模型
    smiles_vocab = load_vocab(cfg["data"]["smiles_vocab"])
    pad_id = smiles_vocab["<pad>"]

    model = ChemG2S(cfg, vocab_size=len(smiles_vocab), pad_id=pad_id).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()

    # 模板资源
    tpl_vocab = load_template_vocab_map(cfg["data"]["template_id_vocab"])
    with open(cfg["data"]["template_roles"], "r", encoding="utf-8") as f:
        template_roles = json.load(f)
    with open("/media/aita4090/YRF/CRAFT-main/data/label_template.json", "r", encoding="utf-8") as f:
        label_templates = json.load(f)

    # 测试集（直接读 src/tgt 文本）
    src_lines = read_lines(args.src_test or cfg["data"].get("src_test"))
    tgt_lines = read_lines(args.tgt_test or cfg["data"].get("tgt_test"))

    # 评测（解码式 Top-n）
    top1, top5, top10, invalid_gold = evaluate_topn(
        cfg, model, smiles_vocab, tpl_vocab, template_roles, label_templates,
        src_lines, tgt_lines, device,
        beam_size=args.beam_size, nbest=args.nbest, max_len=args.max_len, temperature=args.temperature
    )

    print(f"[{human_time()}] TEST(top-n, beam={args.beam_size}) "
          f"top1={top1:.4f} top5={top5:.4f} top10={top10:.4f} invalid_gold={invalid_gold}")

if __name__ == "__main__":
    main()
