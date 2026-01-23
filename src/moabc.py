import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class Food:
    x: np.ndarray
    value: float
    w1: float
    w2: float
    feasible: bool
    trial: int


def _decode(x: np.ndarray, values: np.ndarray, w1s: np.ndarray, w2s: np.ndarray, cap: float):
    val = float(x @ values)
    w1 = float(x @ w1s)
    w2 = float(x @ w2s)
    feasible = (w1 <= cap)
    viol = max(0.0, w1 - cap)
    return val, w1, w2, feasible, viol


def _dominates(a: Dict, b: Dict) -> bool:
    # feasible dominates infeasible
    if a["feasible"] and (not b["feasible"]):
        return True
    if (not a["feasible"]) and b["feasible"]:
        return False
    # both infeasible: smaller violation better
    if (not a["feasible"]) and (not b["feasible"]):
        return a["viol"] < b["viol"]
    # both feasible
    not_worse = (a["value"] >= b["value"]) and (a["w2"] <= b["w2"])
    strictly = (a["value"] > b["value"]) or (a["w2"] < b["w2"])
    return not_worse and strictly


def _pareto_archive_update(archive: List[Dict], candidates: List[Dict], archive_max: int, rng: np.random.Generator) -> List[Dict]:
    combined = archive + candidates
    nd = []
    for i, a in enumerate(combined):
        if not a["feasible"]:
            continue
        dominated = False
        for j, b in enumerate(combined):
            if i == j:
                continue
            if _dominates(b, a):
                dominated = True
                break
        if not dominated:
            nd.append(a)

    # prune if too large: keep diversity by w2 bins
    if len(nd) > archive_max:
        nd = sorted(nd, key=lambda d: d["w2"])
        # sample evenly
        idx = np.linspace(0, len(nd) - 1, archive_max).astype(int)
        nd = [nd[i] for i in idx.tolist()]

    return nd


def _hypervolume_2d(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    pts = points[(points[:, 0] >= 0) & (points[:, 1] >= 0)]
    if pts.size == 0:
        return 0.0
    pts = pts[np.argsort(-pts[:, 0])]
    hv = 0.0
    best_y = 0.0
    for x, y in pts:
        if y > best_y:
            hv += x * (y - best_y)
            best_y = y
    return float(hv)


def _archive_hv(archive: List[Dict], v_sum: float, w2_sum: float) -> float:
    if not archive:
        return 0.0
    vals = np.array([a["value"] for a in archive], dtype=float)
    w2s = np.array([a["w2"] for a in archive], dtype=float)

    v_norm = np.clip(vals / (v_sum if v_sum != 0 else 1.0), 0, 1)
    w_norm = np.clip(w2s / (w2_sum if w2_sum != 0 else 1.0), 0, 1)
    pts = np.column_stack([v_norm, 1.0 - w_norm])
    return _hypervolume_2d(pts)


def _fitness_for_selection(food: Food, cap: float) -> float:
    # prioritize feasible, then higher value, then lower w2
    if food.feasible:
        return food.value - 1e-6 * food.w2
    # penalize infeasible
    return -1e9 - food.w1


def _neighbor_flip(x: np.ndarray, rng: np.random.Generator, flip_rate: float) -> np.ndarray:
    y = x.copy()
    mask = rng.random(y.shape[0]) < flip_rate
    y[mask] = 1 - y[mask]
    # ensure at least one bit flips
    if not mask.any():
        j = rng.integers(0, y.shape[0])
        y[j] = 1 - y[j]
    return y


def run_moabc(
    items_df: pd.DataFrame,
    cap_w1: float,
    colony_size: int,
    food_sources: int,
    cycles: int,
    limit: int,
    flip_rate: float,
    seed: int,
    archive_max: int = 300,
    target_hv: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:

    rng = np.random.default_rng(seed)

    values = items_df["value"].to_numpy(dtype=float)
    w1s = items_df["w1"].to_numpy(dtype=float)
    w2s = items_df["w2"].to_numpy(dtype=float)
    n = len(items_df)

    v_sum = float(values.sum())
    w2_sum = float(w2s.sum())

    # init food sources
    foods: List[Food] = []
    for _ in range(food_sources):
        x = (rng.random(n) < 0.5).astype(np.int8)
        val, w1, w2, feas, _ = _decode(x, values, w1s, w2s, cap_w1)
        foods.append(Food(x=x, value=val, w1=w1, w2=w2, feasible=feas, trial=0))

    archive: List[Dict] = []
    conv_rows = []
    t0 = time.time()
    reached_iter = None

    employed = food_sources
    onlooker = max(0, colony_size - employed)

    for c in range(cycles):
        candidates = []

        # employed bee phase
        for i in range(food_sources):
            new_x = _neighbor_flip(foods[i].x, rng, flip_rate)
            val, w1, w2, feas, viol = _decode(new_x, values, w1s, w2s, cap_w1)

            cand = {"x": new_x, "value": val, "w1": w1, "w2": w2, "feasible": feas, "viol": viol}
            candidates.append(cand)

            # greedy update using dominance on scalar score
            old_score = _fitness_for_selection(foods[i], cap_w1)
            new_score = (val - 1e-6 * w2) if feas else (-1e9 - w1)
            if new_score > old_score:
                foods[i] = Food(x=new_x, value=val, w1=w1, w2=w2, feasible=feas, trial=0)
            else:
                foods[i].trial += 1

        # onlooker phase, probabilistic selection
        scores = np.array([max(1e-9, _fitness_for_selection(f, cap_w1) - (-1e9)) for f in foods], dtype=float)
        probs = scores / scores.sum() if scores.sum() > 0 else np.ones(food_sources) / food_sources

        for _ in range(onlooker):
            i = int(rng.choice(np.arange(food_sources), p=probs))
            new_x = _neighbor_flip(foods[i].x, rng, flip_rate)
            val, w1, w2, feas, viol = _decode(new_x, values, w1s, w2s, cap_w1)

            cand = {"x": new_x, "value": val, "w1": w1, "w2": w2, "feasible": feas, "viol": viol}
            candidates.append(cand)

            old_score = _fitness_for_selection(foods[i], cap_w1)
            new_score = (val - 1e-6 * w2) if feas else (-1e9 - w1)
            if new_score > old_score:
                foods[i] = Food(x=new_x, value=val, w1=w1, w2=w2, feasible=feas, trial=0)
            else:
                foods[i].trial += 1

        # scout phase
        for i in range(food_sources):
            if foods[i].trial >= limit:
                x = (rng.random(n) < 0.5).astype(np.int8)
                val, w1, w2, feas, _ = _decode(x, values, w1s, w2s, cap_w1)
                foods[i] = Food(x=x, value=val, w1=w1, w2=w2, feasible=feas, trial=0)

        # update archive
        archive = _pareto_archive_update(archive, candidates, archive_max, rng)

        # log
        hv = _archive_hv(archive, v_sum, w2_sum)
        best_val = max([a["value"] for a in archive], default=0.0)
        min_w2 = min([a["w2"] for a in archive], default=0.0)

        conv_rows.append(
            {"cycle": c, "hypervolume": hv, "pareto_size": len(archive), "best_value": best_val, "min_w2": min_w2}
        )

        if target_hv is not None and reached_iter is None and hv >= target_hv:
            reached_iter = c

    pareto_df = pd.DataFrame(
        [{"value": a["value"], "w2": a["w2"], "w1": a["w1"]} for a in archive]
    ).sort_values(["w2", "value"], ascending=[True, False], ignore_index=True)

    conv_df = pd.DataFrame(conv_rows)

    summary = {
        "algorithm": "MOABC",
        "seed": seed,
        "colony_size": colony_size,
        "food_sources": food_sources,
        "cycles": cycles,
        "limit": limit,
        "flip_rate": flip_rate,
        "archive_max": archive_max,
        "runtime_seconds": float(time.time() - t0),
        "pareto_size": int(len(pareto_df)),
        "final_hypervolume": float(conv_df["hypervolume"].iloc[-1]) if len(conv_df) else 0.0,
        "reached_target_cycle": reached_iter
    }

    return pareto_df, conv_df, summary
