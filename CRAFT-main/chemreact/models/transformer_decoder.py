# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)

class TransformerDecoderOnly(nn.Module):
    """
    仅Decoder栈（PyTorch TransformerDecoder），memory=图编码H，做交叉注意力。
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dim_ff: int,
                 dropout: float=0.1, tie_weights: bool=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                           dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.lm = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm.weight = self.tok_emb.weight
        self.d_model = d_model

    def _causal_mask(self, L: int, device):
        m = torch.full((L, L), float("-inf"), device=device)
        m = torch.triu(m, diagonal=1)
        return m

    def forward(self, tgt_in_ids, memory, memory_key_padding_mask=None):
        """
        tgt_in_ids: [B,L]
        memory: [B,N,H]
        memory_key_padding_mask: [B,N] (True=pad位置)
        返回 logits: [B,L,V]
        """
        B, L = tgt_in_ids.shape
        x = self.tok_emb(tgt_in_ids) * (self.d_model ** 0.5)
        x = self.pos(x)
        tgt_mask = self._causal_mask(L, tgt_in_ids.device)
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        logits = self.lm(out)
        return logits
