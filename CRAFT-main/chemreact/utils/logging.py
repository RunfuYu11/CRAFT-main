# -*- coding: utf-8 -*-
import time
from typing import Dict

class SmoothedValue:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += int(n)

    @property
    def avg(self):
        return self.sum / max(1, self.count)

def log_metrics(step: int, metrics: Dict[str, float], prefix: str = ""):
    s = f"[{prefix} step={step}] " + " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    print(s)

def human_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
