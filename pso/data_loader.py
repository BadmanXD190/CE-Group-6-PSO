from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

def load_knapsack_instance(csv_path: str, config_path: str):
    items = pd.read_csv(csv_path)
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

    required = {"item_id", "value", "w1", "w2"}
    if not required.issubset(items.columns):
        raise ValueError("CSV must contain item_id, value, w1, w2")

    capacity_w1 = int(cfg["capacity_w1"])
    return items, capacity_w1, cfg

def list_instances(folder: str):
    p = Path(folder)
    csv_files = sorted(p.glob("*.csv"))
    pairs = []
    for csv_f in csv_files:
        cfg_f = p / (csv_f.stem + "_config.json")
        if cfg_f.exists():
            pairs.append((str(csv_f), str(cfg_f)))
    return pairs
