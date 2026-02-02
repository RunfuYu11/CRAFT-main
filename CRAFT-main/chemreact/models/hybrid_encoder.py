import torch
import torch.nn as nn

from .gnn_encoder import GNNEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        import math
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class HybridSeqEncoder(nn.Module):
    """
    图编码(GAT, 带边) + 序列编码(TransformerEncoder)
    融合策略：对齐后的图原子表示与SMILES token嵌入拼接，再线性投影到 d_model。
    """
    def __init__(self,
                 vocab_size: int, d_model: int, pad_id: int,
                 node_in_dim: int, role_dim: int, gnn_proj_dim: int,
                 gnn_hidden: int, gnn_layers: int, gnn_heads: int, gnn_dropout: float,
                 edge_dim: int,
                 n_layers_seq: int = 2, n_heads_seq: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pad_id = pad_id

        # 序列侧
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads_seq,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder_seq = nn.TransformerEncoder(enc_layer, num_layers=n_layers_seq)
        self.pos = PositionalEncoding(d_model, dropout)

        # 图侧（PyG GAT，带边特征）
        self.gnn = GNNEncoder(
            node_in_dim=node_in_dim,
            role_dim=role_dim,
            proj_dim=gnn_proj_dim,
            hidden=gnn_hidden,
            layers=gnn_layers,
            heads=gnn_heads,
            dropout=gnn_dropout,
            edge_dim=edge_dim,
        )

        # 拼接后回投影
        self.concat_proj = nn.Linear(d_model + self.gnn.out_dim, d_model)

    @torch.no_grad()
    def _gather_by_token2atom(self, H_gnn: torch.Tensor, token2atom: torch.Tensor) -> torch.Tensor:
        """
        将原子级表示映射到序列位置；对非原子(-1)位置用0向量。
        H_gnn: [B,N,Hg], token2atom: [B,L]
        return: [B,L,Hg]
        """
        B, L = token2atom.shape
        N = H_gnn.size(1)
        zero_atom = torch.zeros(B, 1, H_gnn.size(-1), device=H_gnn.device, dtype=H_gnn.dtype)
        H_pad = torch.cat([H_gnn, zero_atom], dim=1)      # [B,N+1,Hg]
        idx = token2atom.clamp(min=-1, max=N-1) + 1       # -1->0，其余+1
        idx = idx.unsqueeze(-1).expand(B, L, H_gnn.size(-1))
        gathered = H_pad.gather(dim=1, index=idx)         # [B,L,Hg]
        return gathered

    def forward(self,
                node_feat: torch.Tensor,      # [B,N_max, node_in_dim]
                atom_role: torch.Tensor,      # [B,N_max]（保留原参数，不在本类内做嵌入）
                role_emb: torch.Tensor,       # [B,N_max, role_dim]（由上层 RoleEmbeddingManager 生成）
                node_mask: torch.Tensor,      # [B,N_max]  True=有效
                edge_index: torch.Tensor,     # [2, E_total]
                edge_attr: torch.Tensor,      # [E_total, edge_dim]
                src_ids: torch.Tensor,        # [B,Ls]
                src_pad_mask: torch.Tensor,   # [B,Ls]  True=pad
                token2atom: torch.Tensor      # [B,Ls], -1=非原子
                ):
        # 图侧
        H_gnn = self.gnn(node_feat, role_emb, edge_index, edge_attr, node_mask)   # [B,N,Hg]

        # 序列侧
        X_smi = self.emb(src_ids)                                                 # [B,Ls,d_model]

        # 对齐 + 拼接 + 投影
        H_tok = self._gather_by_token2atom(H_gnn, token2atom)                     # [B,Ls,Hg]
        X_hybrid = torch.cat([X_smi, H_tok], dim=-1)                              # [B,Ls,d_model+Hg]
        X_hybrid = self.concat_proj(X_hybrid)                                     # [B,Ls,d_model]

        # 位置编码 + 编码器
        X = self.pos(X_hybrid)
        memory = self.encoder_seq(X, src_key_padding_mask=src_pad_mask)           # [B,Ls,d_model]
        return memory, src_pad_mask
