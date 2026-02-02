# -*- coding: utf-8 -*-
import json
import torch
import torch.nn as nn

class RoleEmbeddingManager(nn.Module):
    """
    为每个模板维护独立的角色嵌入表（0=非中心）。模板键使用 template_id_vocab.json 的字符串ID。
    前向时依据 batch 的 template_id(int) 找到对应模板字符串ID，选择对应embedding执行索引。
    """
    def __init__(self, template_roles_path: str, template_id_vocab_path: str, role_dim: int):
        super().__init__()
        with open(template_roles_path, "r", encoding="utf-8") as f:
            self.template_roles = json.load(f)
        with open(template_id_vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        # vocab 可能是 {"template_id_to_int": {...}} 或 {tid: int}
        if "template_id_to_int" in vocab:
            id2int = vocab["template_id_to_int"]
            self.tid2int = {str(k): int(v) for k, v in id2int.items()}
        else:
            self.tid2int = {str(k): int(v) for k, v in vocab.items()}
        self.int2tid = {v:k for k,v in self.tid2int.items()}

        modules = nn.ModuleDict()
        for tid_str, info in self.template_roles.items():
            num_roles = int(info.get("num_roles", 1))
            emb = nn.Embedding(num_roles, role_dim)
            nn.init.xavier_uniform_(emb.weight)
            modules[tid_str] = emb
        self.tables = modules
        self.role_dim = role_dim

    def forward(self, template_id_int: torch.LongTensor, atom_role: torch.LongTensor):
        """
        template_id_int: [B]，每个样本的模板ID（int）
        atom_role: [B, N]，每个样本的角色索引（0=非中心）
        返回：role_emb [B, N, role_dim]
        """
        B, N = atom_role.shape
        device = atom_role.device
        out = torch.zeros(B, N, self.role_dim, device=device, dtype=torch.float32)
        for i in range(B):
            tid_int = int(template_id_int[i].item())
            tid_str = self.int2tid[tid_int]
            table = self.tables[tid_str]                       # nn.Embedding
            out[i] = table(atom_role[i])
        return out
