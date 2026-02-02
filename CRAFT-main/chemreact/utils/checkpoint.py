# -*- coding: utf-8 -*-
import os
import torch

def save_checkpoint(path, model, optimizer, scheduler, step, epoch, cfg, vocab):
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "epoch": epoch,
        "cfg": cfg,
        "vocab": vocab
    }
    torch.save(obj, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    obj = torch.load(path, map_location=map_location)
    model.load_state_dict(obj["model"])
    if optimizer is not None and obj.get("optimizer") is not None:
        optimizer.load_state_dict(obj["optimizer"])
    if scheduler is not None and obj.get("scheduler") is not None:
        scheduler.load_state_dict(obj["scheduler"])
    return obj
