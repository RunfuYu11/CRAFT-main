# -*- coding: utf-8 -*-
from typing import Tuple
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdchem

# 原子特征向量上限；不足零填充、超过截断
ATOM_FEAT_DIM = 32  # 可由配置覆盖

def _one_hot_bucket(val: int, choices):
    v = [0]*len(choices)
    try:
        idx = choices.index(val)
    except ValueError:
        idx = len(choices)-1
    v[idx] = 1
    return v

def atom_features(a: rdchem.Atom):
    """
    原子特征（更新）:
      - 原子类型索引: a.GetAtomicNum()（标量，替代原one-hot）
      - 芳香性: 0/1
      - 成环: 0/1
      - 形式电荷: one-hot [-2,-1,0,1,2]（与原逻辑一致）
      - 原子度数: one-hot [0,1,2,3,4,其他]（与原逻辑一致）
      - 手性标志: HasProp("_CIPCode")
      - 杂化态: one-hot [sp, sp2, sp3, other]（与原逻辑一致）
      - 总氢数: a.GetTotalNumHs()（标量）
      - 总价态: a.GetTotalValence()（标量）
    最终维度 ≈ 21，后续按 ATOM_FEAT_DIM 进行零填/截断。
    """
    z = a.GetAtomicNum()  # 标量
    aromatic = [int(a.GetIsAromatic())]
    in_ring = [int(a.IsInRing())]
    chg = _one_hot_bucket(a.GetFormalCharge(), [-2, -1, 0, 1, 2])
    deg = _one_hot_bucket(a.GetDegree(), [0, 1, 2, 3, 4, 99])
    chiral = [int(a.HasProp("_CIPCode"))]
    hyb_map = {
        rdchem.HybridizationType.SP: 0,
        rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2
    }
    hyb = _one_hot_bucket(hyb_map.get(a.GetHybridization(), 3), [0, 1, 2, 3])
    total_h = [int(a.GetTotalNumHs())]
    total_val = [int(a.GetTotalValence())]

    v = [float(z)] + aromatic + in_ring + chg + deg + chiral + hyb + total_h + total_val
    return np.array(v, dtype=np.float32)

def bond_features(b: rdchem.Bond):
    """
    键特征:
      - 键类型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC
      - 共轭: 0/1
      - 成环: 0/1
    """
    bt = b.GetBondType()
    bt_choices = [
        rdchem.BondType.SINGLE,
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
        rdchem.BondType.AROMATIC,
    ]
    bt_oh = _one_hot_bucket(bt, bt_choices)
    conj = [int(b.GetIsConjugated())]
    ring = [int(b.IsInRing())]
    v = bt_oh + conj + ring  # 4 + 1 + 1 = 6
    return np.array(v, dtype=np.float32)

def smiles_to_graph(smiles: str, atom_feat_dim_cfg: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
      X: [n, atom_feat_dim_cfg]  原子特征(零填/截断到上限)
      edge_index: [2, E]         无向图转成双向有向边
      edge_attr: [E, edge_dim]   边特征（6维）
    注意: 不手动加自环，GATConv(add_self_loops=True)会处理
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit parse failed: {smiles}")

    n = mol.GetNumAtoms()
    # 原子特征
    feats = [atom_features(mol.GetAtomWithIdx(i)) for i in range(n)]
    fdim = len(feats[0]) if n > 0 else 0
    X = np.zeros((n, atom_feat_dim_cfg), dtype=np.float32)
    for i, feat in enumerate(feats):
        if feat.shape[0] >= atom_feat_dim_cfg:
            X[i, :] = feat[:atom_feat_dim_cfg]
        else:
            X[i, :feat.shape[0]] = feat

    # 边
    ei = []
    ea = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bf = bond_features(b)
        # 无向→双向
        ei.append([i, j]); ea.append(bf)
        ei.append([j, i]); ea.append(bf)
    edge_index = np.array(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.stack(ea, axis=0).astype(np.float32) if ea else np.zeros((0, 6), dtype=np.float32)

    return X, edge_index, edge_attr
