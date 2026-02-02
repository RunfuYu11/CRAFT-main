import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from chemreact.utils.config import load_config, parse_args, override_cfg, ensure_dirs
from chemreact.utils.checkpoint import save_checkpoint
from chemreact.utils.logging import human_time
from chemreact.data.tokenization import load_vocab
from chemreact.data.dataset import ChemGenDataset
from chemreact.data.collate import collate_fn
from chemreact.models.chem_g2s import ChemG2S

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def compute_batch_metrics(logits: torch.Tensor, tgt_out: torch.Tensor, pad_id: int = 0):
    """
    logits: [B,L,V]
    tgt_out: [B,L]
    返回：token_acc(标量), perfect_acc(标量), token_num(用于加权)
    """
    pred = logits.argmax(dim=-1)                  # [B,L]
    mask = (tgt_out != pad_id)                    # [B,L]
    correct = (pred == tgt_out) & mask            # [B,L]
    token_num = mask.sum().item()
    token_correct = correct.sum().item()
    token_acc = (token_correct / token_num) if token_num > 0 else 0.0

    # perfect acc：序列所有非pad位置全对
    per_sample_valid = mask.sum(dim=1) > 0
    per_sample_all_correct = (correct | ~mask).all(dim=1) & per_sample_valid
    denom = per_sample_valid.sum().item()
    perfect_acc = (per_sample_all_correct.sum().item() / denom) if denom > 0 else 0.0
    return token_acc, perfect_acc, token_num

@torch.no_grad()
def evaluate(model, loader, criterion, device, amp, pad_id=0):
    model.eval()
    loss_sum = 0.0
    sample_sum = 0
    token_correct_sum = 0
    token_total_sum = 0
    perfect_correct_sum = 0
    perfect_total_sum = 0

    tensor_keys = [
        "template_id",
        "node_feat", "atom_role", "node_mask",
        "pyg_edge_index", "pyg_edge_attr",
        "src_ids", "src_pad_mask", "token2atom",
        "tgt_in", "tgt_out", "tgt_pad_mask",
    ]

    for batch in loader:
        for k in tensor_keys:
            batch[k] = batch[k].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
            logits = model(batch)  # [B,L,V]
            B, L, V = logits.shape
            loss = criterion(logits.reshape(B*L, V), batch["tgt_out"].reshape(B*L))

        loss_sum += float(loss.item()) * B
        sample_sum += int(B)

        tok_acc, perf_acc, tok_cnt = compute_batch_metrics(logits, batch["tgt_out"], pad_id=pad_id)
        token_correct_sum += tok_acc * tok_cnt
        token_total_sum  += tok_cnt

        perfect_correct_sum += perf_acc * B
        perfect_total_sum  += B

    avg_loss = loss_sum / max(1, sample_sum)
    avg_tok_acc = token_correct_sum / max(1, token_total_sum)
    avg_perf_acc = perfect_correct_sum / max(1, perfect_total_sum)
    return avg_loss, avg_tok_acc, avg_perf_acc

def train_main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = override_cfg(cfg, args)
    ensure_dirs(cfg)

    # 设备
    dev_name = cfg["misc"]["device"]
    if dev_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_name)
    print(f"[{human_time()}] device={device}")

    # 种子
    set_seed(cfg["train"]["seed"])

    # 词表
    vocab = load_vocab(cfg["data"]["smiles_vocab"])
    vocab_size = len(vocab)
    pad_id = 0  # SPECIALS 定义已保证

    # 数据
    train_ds = ChemGenDataset(
        src_path=cfg["data"]["src_train"],
        tgt_path=cfg["data"]["tgt_train"],
        enc_path=cfg["data"]["enc_train"],
        smiles_vocab_path=cfg["data"]["smiles_vocab"],
        atom_feat_dim=cfg["model"]["atom_feat_dim"]
    )
    valid_ds = ChemGenDataset(
        src_path=cfg["data"]["src_valid"],
        tgt_path=cfg["data"]["tgt_valid"],
        enc_path=cfg["data"]["enc_valid"],
        smiles_vocab_path=cfg["data"]["smiles_vocab"],
        atom_feat_dim=cfg["model"]["atom_feat_dim"]
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["misc"]["num_workers"], pin_memory=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["misc"]["num_workers"], pin_memory=True, collate_fn=collate_fn
    )

    # 模型
    model = ChemG2S(cfg, vocab_size=vocab_size, pad_id=pad_id).to(device)

    # 优化与调度
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=tuple(cfg["train"]["betas"]),
        weight_decay=cfg["train"]["weight_decay"]
    )
    pl_cfg = cfg["train"]["plateau"]
    scheduler = ReduceLROnPlateau(
        optim,
        mode="min",
        factor=pl_cfg["factor"],
        patience=pl_cfg["patience"],
        threshold=pl_cfg["threshold"],
        cooldown=pl_cfg["cooldown"],
        min_lr=pl_cfg["min_lr"],
        verbose=True
    )

    # 损失
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=cfg["train"]["label_smoothing"])

    # AMP
    amp = bool(cfg["train"]["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    # 早停
    patience = cfg["train"]["early_stop_patience"]
    best_val = float("inf")
    best_epoch = -1
    wait = 0

    global_step = 0
    max_epochs = cfg["train"]["max_epochs"]

    tensor_keys = [
        "template_id",
        "node_feat", "atom_role", "node_mask",
        "pyg_edge_index", "pyg_edge_attr",
        "src_ids", "src_pad_mask", "token2atom",
        "tgt_in", "tgt_out", "tgt_pad_mask",
    ]

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_sum = 0
        tok_correct_sum = 0
        tok_total_sum = 0
        perf_correct_sum = 0
        perf_total_sum = 0

        pbar = tqdm(train_loader, total=len(train_loader), ncols=150,
                    desc=f"epoch {epoch}/{max_epochs} train")

        for batch in pbar:
            for k in tensor_keys:
                batch[k] = batch[k].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits = model(batch)  # [B,L,V]
                B, L, V = logits.shape
                loss = criterion(logits.reshape(B*L, V), batch["tgt_out"].reshape(B*L))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optim)
            scaler.update()

            # 统计
            epoch_loss_sum += float(loss.item()) * B
            epoch_sample_sum += int(B)

            tok_acc, perf_acc, tok_cnt = compute_batch_metrics(logits, batch["tgt_out"], pad_id=pad_id)
            tok_correct_sum += tok_acc * tok_cnt
            tok_total_sum  += tok_cnt
            perf_correct_sum += perf_acc * B
            perf_total_sum  += B

            global_step += 1
            pbar.set_postfix({
                "loss": f"{(epoch_loss_sum/max(1,epoch_sample_sum)):.4f}",
                "tok_acc": f"{(tok_correct_sum/max(1,tok_total_sum)):.4f}",
                "perf_acc": f"{(perf_correct_sum/max(1,perf_total_sum)):.4f}"
            })

        # 训练epoch指标
        train_loss = epoch_loss_sum / max(1, epoch_sample_sum)
        train_tok_acc = tok_correct_sum / max(1, tok_total_sum)
        train_perf_acc = perf_correct_sum / max(1, perf_total_sum)

        # 验证
        val_loss, val_tok_acc, val_perf_acc = evaluate(
            model, valid_loader, criterion, device, amp, pad_id=pad_id
        )

        # 调度与日志
        scheduler.step(val_loss)
        cur_lr = optim.param_groups[0]["lr"]
        print(f"[{human_time()}] epoch={epoch} "
              f"train_loss={train_loss:.4f} train_tok_acc={train_tok_acc:.4f} train_perf_acc={train_perf_acc:.4f} | "
              f"valid_loss={val_loss:.4f} valid_tok_acc={val_tok_acc:.4f} valid_perf_acc={val_perf_acc:.4f} | "
              f"lr={cur_lr:.6g}")

        # 早停/保存
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
            path = os.path.join(cfg["train"]["ckpt_dir"], "best.pt")
            save_checkpoint(path, model, optim, scheduler, step=global_step, epoch=epoch, cfg=cfg, vocab=vocab)
        else:
            wait += 1
            if wait >= patience:
                print(f"[{human_time()}] early stopping at epoch={epoch}, best_epoch={best_epoch}, best_val={best_val:.4f}")
                break

if __name__ == "__main__":
    train_main()
