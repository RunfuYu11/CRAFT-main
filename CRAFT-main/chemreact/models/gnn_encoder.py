# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # x: [N, D]
        return self.net(x)

class GNNEncoder(nn.Module):
    """
    PyG GAT 编码器（带边特征）：
      输入:
        - node_feat: [B,N,d_node]（padding 后）
        - role_emb:  [B,N,d_role]（与节点对齐）
        - edge_index: [2, E_total]（按 batch 拼接后的全局索引）
        - edge_attr:  [E_total, d_edge]
        - node_mask:  [B,N]  True=有效
      输出:
        - H: [B,N,hidden]
    """
    def __init__(self, node_in_dim: int, role_dim: int, proj_dim: int,
                 hidden: int, layers: int, heads: int, dropout: float,
                 edge_dim: int):
        super().__init__()
        self.fuse = nn.Linear(node_in_dim + role_dim, proj_dim)
        self.layers = nn.ModuleList()
        in_dim = proj_dim
        for _ in range(layers):
            conv = GATConv(
                in_channels=in_dim,
                out_channels=hidden // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,          # 关键：使用边特征
                add_self_loops=True,
            )
            self.layers.append(nn.ModuleList([conv, FeedForward(hidden, hidden*4, dropout)]))
            in_dim = hidden
        self.out_dim = hidden

    def forward(self, node_feat, role_emb, edge_index, edge_attr, node_mask):
        device = node_feat.device
        B, Nmax, _ = node_feat.shape

        # 仅取有效节点并展平到 [N_total, *]
        n_list = node_mask.sum(dim=1).tolist()
        index_flat = []
        base = 0
        for b, n in enumerate(n_list):
            row = b * Nmax + torch.arange(n, device=device)
            index_flat.append(row)
        index_flat = torch.cat(index_flat, dim=0) if len(index_flat) > 0 else torch.tensor([], dtype=torch.long, device=device)

        x = torch.cat([node_feat, role_emb], dim=-1)      # [B,Nmax,d_node+d_role]
        x = x.reshape(B * Nmax, -1)[index_flat]           # [N_total, d_node+d_role]
        x = self.fuse(x)                                  # [N_total, proj_dim]

        for conv, ffn in self.layers:
            x = conv(x, edge_index, edge_attr)            # [N_total, hidden]
            x = ffn(x)                                    # [N_total, hidden]

        # 回填到 [B,Nmax,hidden]（pad位置为0）
        H = torch.zeros(B, Nmax, self.out_dim, device=device, dtype=x.dtype)
        start = 0
        for b, n in enumerate(n_list):
            if n > 0:
                H[b, :n] = x[start:start+n]
                start += n
        return H
