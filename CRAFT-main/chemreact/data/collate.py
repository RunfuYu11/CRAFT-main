# -*- coding: utf-8 -*-
from typing import List, Dict, Any
import torch

def collate_fn(batch: List[Dict[str, Any]]):
    B = len(batch)

    # ------- 节点 padding -------
    n_list = [b["node_feat"].shape[0] for b in batch]
    N_max = max(n_list)
    d = batch[0]["node_feat"].shape[1]

    node_feat_b = torch.zeros(B, N_max, d, dtype=batch[0]["node_feat"].dtype)
    atom_role_b = torch.zeros(B, N_max, dtype=torch.long)
    node_mask_b = torch.zeros(B, N_max, dtype=torch.bool)

    # ------- PyG 边 -------
    edge_index_list = []
    edge_attr_list = []
    offset = 0
    edge_dim = None
    for b in range(B):
        n = n_list[b]
        node_feat_b[b, :n] = batch[b]["node_feat"]
        atom_role_b[b, :n] = batch[b]["atom_role"]
        node_mask_b[b, :n] = True

        if "edge_attr" in batch[b] and edge_dim is None:
            edge_dim = batch[b]["edge_attr"].shape[1] if batch[b]["edge_attr"].ndim == 2 else 0
        if "edge_index" in batch[b] and batch[b]["edge_index"].numel() > 0:
            ei = batch[b]["edge_index"] + offset
            edge_index_list.append(ei)
            edge_attr_list.append(batch[b]["edge_attr"])
        offset += n
    if len(edge_index_list) > 0:
        pyg_edge_index = torch.cat(edge_index_list, dim=1)   # [2, E_total]
        pyg_edge_attr = torch.cat(edge_attr_list, dim=0)     # [E_total, edge_dim]
    else:
        pyg_edge_index = torch.zeros(2, 0, dtype=torch.long)
        pyg_edge_attr = torch.zeros(0, edge_dim or 0)

    # ------- 序列 -------
    Ls_list = [b["src_ids"].shape[0] for b in batch]
    Ls_max = max(Ls_list)
    src_ids_b = torch.zeros(B, Ls_max, dtype=batch[0]["src_ids"].dtype)
    src_pad_mask_b = torch.ones(B, Ls_max, dtype=torch.bool)  # True=pad
    token2atom_b = torch.full((B, Ls_max), -1, dtype=torch.long)

    for b in range(B):
        Ls = Ls_list[b]
        src_ids_b[b, :Ls] = batch[b]["src_ids"]
        src_pad_mask_b[b, :Ls] = False
        token2atom_b[b, :Ls] = batch[b]["token2atom"]

    # ------- 目标序列 -------
    Lt_list = [b["tgt_in"].shape[0] for b in batch]
    Lt_max = max(Lt_list)
    tgt_in_b = torch.zeros(B, Lt_max, dtype=batch[0]["tgt_in"].dtype)
    tgt_out_b = torch.zeros(B, Lt_max, dtype=batch[0]["tgt_out"].dtype)
    tgt_pad_mask_b = torch.ones(B, Lt_max, dtype=torch.bool)
    for b in range(B):
        Lt = Lt_list[b]
        tgt_in_b[b, :Lt] = batch[b]["tgt_in"]
        tgt_out_b[b, :Lt] = batch[b]["tgt_out"]
        tgt_pad_mask_b[b, :Lt] = False

    # ------- 模板ID -------
    tpl_ids = []
    for b in range(B):
        tid = batch[b].get("template_id", None)
        if tid is None:
            # 兼容：有的样本可能给 "template_ids"
            tid = batch[b].get("template_ids", None)
        if isinstance(tid, torch.Tensor):
            tid = int(tid.item()) if tid.ndim == 0 else int(tid.view(-1)[0].item())
        if tid is None:
            raise KeyError("样本缺少 template_id/template_ids")
        tpl_ids.append(tid)
    template_id_b = torch.tensor(tpl_ids, dtype=torch.long)

    meta = {
        "src_tokens": [b["src_tokens"] for b in batch],
        "tgt_tokens": [b["tgt_tokens"] for b in batch],
        "src_smi": [b["src_smi"] for b in batch],
    }

    return {
        "template_id": template_id_b,         # 新增 [B]
        "node_feat": node_feat_b,             # [B,N_max,d]
        "atom_role": atom_role_b,             # [B,N_max]
        "node_mask": node_mask_b,             # [B,N_max]
        "pyg_edge_index": pyg_edge_index,     # [2,E_total]
        "pyg_edge_attr": pyg_edge_attr,       # [E_total, edge_dim]
        "src_ids": src_ids_b,                 # [B,Ls_max]
        "src_pad_mask": src_pad_mask_b,       # [B,Ls_max]
        "token2atom": token2atom_b,           # [B,Ls_max]
        "tgt_in": tgt_in_b,
        "tgt_out": tgt_out_b,
        "tgt_pad_mask": tgt_pad_mask_b,
        "meta": meta
    }
