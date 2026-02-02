# -*- coding: utf-8 -*-
import argparse
import os
import yaml

def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="/media/aita4090/YRF/CRAFT-main/configs/default.yaml")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()
    return args

def override_cfg(cfg, args):
    if args.device is not None:
        cfg["misc"]["device"] = args.device
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    return cfg

def ensure_dirs(cfg):
    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
