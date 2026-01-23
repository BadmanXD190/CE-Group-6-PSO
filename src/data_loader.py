import json
import pandas as pd
from typing import Tuple, Dict


def load_instance(dataset_dir: str, inst_name: str) -> Tuple[pd.DataFrame, Dict]:
    csv_path = f"{dataset_dir}/{inst_name}.csv"
    json_path = f"{dataset_dir}/{inst_name}_config.json"

    items = pd.read_csv(csv_path)
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_cols = {"item_id", "value", "w1", "w2"}
    missing = required_cols - set(items.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    if "capacity_w1" not in meta:
        raise ValueError("JSON missing capacity_w1")

    for c in ["value", "w1", "w2"]:
        items[c] = pd.to_numeric(items[c], errors="coerce")
    if items[["value", "w1", "w2"]].isna().any().any():
        raise ValueError("CSV has invalid numeric values")

    return items, meta
