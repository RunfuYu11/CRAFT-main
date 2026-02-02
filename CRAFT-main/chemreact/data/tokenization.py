# -*- coding: utf-8 -*-
import json
from typing import List, Dict

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]  # pad 固定为 id=0

class SmilesTokenizer:
    """词级 tokenizer：输入为已经按空格分好的 token 列表"""
    def __init__(self, vocab: Dict[str, int]):
        self.stoi = vocab
        self.itos = {i: s for s, i in vocab.items()}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    def encode_tokens(self, tokens: List[str], add_bos_eos: bool = True) -> List[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        if add_bos_eos:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode_ids(self, ids: List[int], remove_special: bool = True) -> List[str]:
        out = []
        for i in ids:
            tok = self.itos.get(int(i), "<unk>")
            if remove_special and tok in SPECIALS:
                continue
            out.append(tok)
        return out  # 返回 token 列表

def save_vocab(path: str, vocab: Dict[str, int]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
