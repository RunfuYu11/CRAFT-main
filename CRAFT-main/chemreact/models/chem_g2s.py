# v4/chemreact/models/chem_g2s.py
# -*- coding: utf-8 -*-
import json
import torch
import torch.nn as nn

from .role_embedding import RoleEmbeddingManager
from .hybrid_encoder import HybridSeqEncoder
from .transformer_decoder import TransformerDecoderOnly

class ChemG2S(nn.Module):
    """
    编码端：HybridSeqEncoder（PyG-GAT带边 + 序列Transformer；拼接+线性投影）
    记忆首位：模板全局嵌入（template embedding）
    解码端：Transformer decoder-only
    """
    def __init__(self, cfg, vocab_size: int, pad_id: int):
        super().__init__()
        mcfg = cfg["model"]
        d_model = mcfg["dec_d_model"]

        # 角色嵌入管理（模板依赖）
        self.role_mgr = RoleEmbeddingManager(
            template_roles_path=cfg["data"]["template_roles"],
            template_id_vocab_path=cfg["data"]["template_id_vocab"],
            role_dim=mcfg["role_dim"]
        )

        # 模板全局嵌入
        num_tpl = len(self.role_mgr.int2tid)
        tpl_dim = mcfg["tpl_dim"]
        self.tpl_emb = nn.Embedding(num_tpl, tpl_dim)
        self.tpl_dropout = nn.Dropout(mcfg.get("tpl_dropout", 0.0))
        self.tpl_proj = nn.Linear(tpl_dim, d_model) if tpl_dim != d_model else nn.Identity()
        self.tpl_norm = nn.LayerNorm(d_model)

        # 编码器（图+序列）
        self.encoder = HybridSeqEncoder(
            vocab_size=vocab_size, d_model=d_model, pad_id=pad_id,
            node_in_dim=mcfg["atom_feat_dim"], role_dim=mcfg["role_dim"], gnn_proj_dim=mcfg["proj_dim"],
            gnn_hidden=mcfg["gnn_hidden"], gnn_layers=mcfg["gnn_layers"], gnn_heads=mcfg["gat_heads"],
            gnn_dropout=mcfg["gat_dropout"], edge_dim=mcfg["edge_dim"],
            n_layers_seq=mcfg["enc_num_layers"], n_heads_seq=mcfg["enc_nhead"],
            ff_mult=mcfg["enc_ffn"] // d_model, dropout=mcfg["enc_dropout"]
        )

        # 解码器
        self.decoder = TransformerDecoderOnly(
            vocab_size=vocab_size, d_model=d_model, nhead=mcfg["dec_nhead"],
            num_layers=mcfg["dec_num_layers"], dim_ff=mcfg["dec_ffn"],
            dropout=mcfg["dec_dropout"], tie_weights=bool(mcfg.get("tie_weights", True))
        )

    def _get_tpl_id(self, batch):
        # 兼容多命名
        if "template_id" in batch:
            return batch["template_id"].long()
        if "tpl_id" in batch:
            return batch["tpl_id"].long()
        if "template_ids" in batch:  # 有的dataset可能直接给batch级别向量
            return batch["template_ids"].long()
        raise KeyError("batch 中未找到模板ID（template_id / tpl_id / template_ids）")

    def forward(self, batch):
        # 准备输入
        node_feat   = batch["node_feat"]            # [B,N,d_node]
        atom_role   = batch["atom_role"]            # [B,N]
        node_mask   = batch["node_mask"]            # [B,N] True=有效
        edge_index  = batch["pyg_edge_index"]       # [2,E_total]
        edge_attr   = batch["pyg_edge_attr"]        # [E_total,d_edge]

        src_ids     = batch["src_ids"]              # [B,Ls]
        src_pad_mask= batch["src_pad_mask"]         # [B,Ls] True=pad
        token2atom  = batch["token2atom"]           # [B,Ls]

        tgt_in      = batch["tgt_in"]               # [B,Lt]

        # 角色嵌入（模板依赖）
        tpl_id = self._get_tpl_id(batch)            # [B]
        role_emb = self.role_mgr(tpl_id, atom_role) # [B,N,role_dim]

        # 编码（图+序列）
        memory_seq, mem_pad_seq = self.encoder(
            node_feat=node_feat, atom_role=atom_role, role_emb=role_emb, node_mask=node_mask,
            edge_index=edge_index, edge_attr=edge_attr,
            src_ids=src_ids, src_pad_mask=src_pad_mask, token2atom=token2atom
        )                                           # [B,Ls,d_model], [B,Ls]

        # 模板全局token（置于memory首位）
        e_tpl = self.tpl_emb(tpl_id)                # [B,tpl_dim]
        e_tpl = self.tpl_proj(e_tpl)                # → [B,d_model]
        e_tpl = self.tpl_dropout(e_tpl)
        e_tpl = self.tpl_norm(e_tpl).unsqueeze(1)   # [B,1,d_model]

        memory = torch.cat([e_tpl, memory_seq], dim=1)        # [B,1+Ls,d_model]
        B = memory.size(0)
        tpl_mask = torch.zeros(B, 1, dtype=torch.bool, device=memory.device)  # False=有效
        mem_pad_mask = torch.cat([tpl_mask, mem_pad_seq], dim=1)              # [B,1+Ls]

        # 解码
        logits = self.decoder(tgt_in, memory, memory_key_padding_mask=mem_pad_mask)  # [B,Lt,V]
        return logits
