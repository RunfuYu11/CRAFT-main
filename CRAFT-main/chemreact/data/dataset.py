# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Dict, Any, List

from .tokenization import SmilesTokenizer, load_vocab
from .rdkit_ops import smiles_to_graph
from rdkit import Chem

def _read_lines(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def _parse_src_line(line: str) -> Tuple[str, List[str], str]:
    """
    输入: "[91] C C 1 ( C ) ..."（已分词）
    返回: (模板ID字符串, 源token列表, 去空格SMILES)
    """
    parts = line.strip().split()
    import re
    m = re.search(r"\d+", parts[0])
    tid = m.group(0)
    tokens = parts[1:]
    smi = "".join(tokens)
    return tid, tokens, smi

def _load_enc(path: Path) -> Dict[str, Any]:
    p = str(path)
    if p.endswith(".npz"):
        data = np.load(p, allow_pickle=True)
        return {
            "atom_role_flat": data["atom_role_flat"],
            "row_ptr": data["row_ptr"],
            "template_ids": data["template_ids"],
            "comp_idx": data.get("comp_idx", None),
            "mode": data.get("mode", None),
            "line_numbers": data.get("line_numbers", None),
        }
    else:
        obj = np.load(p, allow_pickle=True).item()
        return obj

# -------- 原子 token 判定（按你给定词表）--------
# 多字符元素：在你的词表中存在 Br、Cl；Si 不在（仅有 "[Si]" 这类括号原子），故不计 "Si"
_ATOM_MULTI = {"Br", "Cl"}
# 单字符大写元素：B C N O P S F I（与你的词表一致）
_ATOM_UP = {"B", "C", "N", "O", "P", "S", "F", "I"}
# 芳香小写：c n o s
_ATOM_LOW = {"c", "n", "o", "s"}

def _is_atom_token(tok: str) -> bool:
    if not tok:
        return False
    # 括号原子（如 [N+], [Si], [nH] ...）
    if tok.startswith("[") and tok.endswith("]"):
        return True
    if tok in _ATOM_MULTI:
        return True
    if tok in _ATOM_UP:
        return True
    if tok in _ATOM_LOW:
        return True
    return False

def _build_token2atom(tokens: List[str], mol: Chem.Mol) -> np.ndarray:
    """
    顺序对齐启发式：
      - 按 tokens 中原子 token 的出现顺序与 RDKit 原子顺序一一对应
      - 非原子位置填 -1
    """
    L = len(tokens)
    n_atoms = mol.GetNumAtoms()
    arr = np.full((L,), -1, dtype=np.int64)
    atom_pos = [i for i, t in enumerate(tokens) if _is_atom_token(t)]
    K = min(len(atom_pos), n_atoms)
    for k in range(K):
        arr[atom_pos[k]] = k
    return arr

class ChemGenDataset(Dataset):
    def __init__(self,
                 src_path: str,
                 tgt_path: str,
                 enc_path: str,
                 smiles_vocab_path: str,
                 atom_feat_dim: int):
        self.src_lines = _read_lines(Path(src_path))
        self.tgt_lines = _read_lines(Path(tgt_path))
        assert len(self.src_lines) == len(self.tgt_lines), "src/tgt 行数不一致"
        self.N = len(self.src_lines)
        self.enc = _load_enc(Path(enc_path))

        self.row_ptr = self.enc["row_ptr"].astype(np.int64)
        self.atom_role_flat = self.enc["atom_role_flat"].astype(np.int32)
        self.template_ids = self.enc["template_ids"].astype(np.int32)

        vocab = load_vocab(smiles_vocab_path)
        self.tok = SmilesTokenizer(vocab)
        self.atom_feat_dim = atom_feat_dim
        self.pad_id = self.tok.pad_id

    def __len__(self):
        return self.N

    def _get_roles(self, idx: int) -> torch.LongTensor:
        start = int(self.row_ptr[idx])
        end = int(self.row_ptr[idx + 1])
        roles = self.atom_role_flat[start:end]
        return torch.from_numpy(roles.astype(np.int64))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tid_str, src_tokens, src_smi = _parse_src_line(self.src_lines[idx])
        tgt_tokens = self.tgt_lines[idx].strip().split()

        # 分子与图
        mol = Chem.MolFromSmiles(src_smi)
        if mol is None:
            raise ValueError(f"RDKit parse failed: {src_smi}")
        X, EI, EA = smiles_to_graph(src_smi, self.atom_feat_dim)
        X = torch.from_numpy(X)  # [n, d]
        EI = torch.from_numpy(EI)  # [2, E]
        EA = torch.from_numpy(EA)  # [E, edge_dim]
        n_atoms = X.shape[0]

        # 角色
        atom_role = self._get_roles(idx)    # [n]
        assert atom_role.shape[0] == n_atoms, f"encoder_input 原子数与 SMILES 解析不一致 idx={idx}"

        tpl_id_int = int(self.template_ids[idx])

        # 源序列 ids（不加 BOS/EOS）
        src_ids = torch.tensor(self.tok.encode_tokens(src_tokens, add_bos_eos=False), dtype=torch.long)  # [Ls]
        Ls = src_ids.shape[0]

        # token→atom 映射（非原子为 -1）
        token2atom_np = _build_token2atom(src_tokens, mol)
        token2atom = torch.from_numpy(token2atom_np)  # [Ls]

        # 目标序列 ids（训练用，含 BOS/EOS）
        tgt_ids = self.tok.encode_tokens(tgt_tokens, add_bos_eos=True)
        tgt = torch.tensor(tgt_ids, dtype=torch.long)
        tgt_in = tgt[:-1]
        tgt_out = tgt[1:]

        return {
            "template_id": tpl_id_int,
            "node_feat": X,
            "edge_index": EI,
            "edge_attr": EA,
            "atom_role": atom_role,
            "node_mask": torch.ones(n_atoms, dtype=torch.bool),

            "src_ids": src_ids,                            # [Ls]
            "token2atom": token2atom,                      # [Ls], -1 表非原子
            "src_pad_mask": torch.zeros(Ls, dtype=torch.bool),  # 单样本先不pad

            "tgt_in": tgt_in,
            "tgt_out": tgt_out,

            "src_tokens": src_tokens,
            "tgt_tokens": tgt_tokens,
            "src_smi": src_smi,
        }
