from __future__ import annotations

import time
import numpy as np
import pandas as pd

from pso.pareto_archive import ParetoArchive, dominates

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def repair_w1_capacity(x: np.ndarray, w1: np.ndarray, capacity: int, rng: np.random.Generator) -> np.ndarray:
    """
    Ensure sum(w1 * x) <= capacity by removing selected items.
    Remove items with worst value-to-w1 ratio first for better repairs.
    """
    total_w1 = float(np.dot(x, w1))
    if total_w1 <= capacity:
        return x

    ones = np.where(x == 1)[0]
    if len(ones) == 0:
        return x

    ratio = np.zeros_like(w1, dtype=float)
    ratio[w1 > 0] = 1.0 / w1[w1 > 0]
    drop_order = ones[np.argsort(ratio[ones])]  # smallest 1/w1 means largest w1
    for idx in drop_order:
        x[idx] = 0
        total_w1 -= w1[idx]
        if total_w1 <= capacity:
            break
    return x

def evaluate_population(X: np.ndarray, value: np.ndarray, w1: np.ndarray, w2: np.ndarray, capacity: int):
    """
    Returns
    raw metrics and minimization objectives F
    F is two columns
    f1 is -total_value
    f2 is total_w2
    """
    total_value = X @ value
    total_w1 = X @ w1
    total_w2 = X @ w2

    feasible = total_w1 <= capacity

    # All solutions should be feasible after repair.
    # If any are not feasible, we apply a penalty to objectives.
    penalty = np.maximum(0.0, total_w1 - capacity)
    f1 = -total_value + 1e6 * penalty
    f2 = total_w2 + 1e6 * penalty

    F = np.vstack([f1, f2]).T
    return total_value, total_w1, total_w2, feasible, F

def update_personal_best(P: np.ndarray, PF: np.ndarray, X: np.ndarray, F: np.ndarray, rng: np.random.Generator):
    """
    Update pbest using Pareto dominance
    If neither dominates, choose one randomly to maintain diversity.
    """
    n = X.shape[0]
    for i in range(n):
        a = F[i]
        b = PF[i]
        if dominates(a, b):
            P[i] = X[i].copy()
            PF[i] = a.copy()
        elif dominates(b, a):
            pass
        else:
            if rng.random() < 0.5:
                P[i] = X[i].copy()
                PF[i] = a.copy()

def mopso_knapsack(
    items: pd.DataFrame,
    capacity_w1: int,
    n_particles: int = 80,
    iters: int = 300,
    w: float = 0.7,
    c1: float = 1.6,
    c2: float = 1.6,
    vmax: float = 6.0,
    archive_size: int = 200,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    value = items["value"].to_numpy(dtype=float)
    w1_arr = items["w1"].to_numpy(dtype=float)
    w2_arr = items["w2"].to_numpy(dtype=float)

    n_bits = len(items)

    start = time.time()

    X = rng.integers(0, 2, size=(n_particles, n_bits), dtype=int)
    V = rng.uniform(-1.0, 1.0, size=(n_particles, n_bits))

    for i in range(n_particles):
        X[i] = repair_w1_capacity(X[i], w1_arr, capacity_w1, rng)

    total_value, total_w1, total_w2, feasible, F = evaluate_population(X, value, w1_arr, w2_arr, capacity_w1)

    P = X.copy()
    PF = F.copy()

    archive = ParetoArchive(max_size=archive_size)
    archive.add(X, F)

    history = []
    # Track best archive point by value for a simple progress signal
    # Since f1 is -value, best value means smallest f1
    best_value_so_far = float(np.max(total_value))
    history.append(best_value_so_far)

    for t in range(iters):
        guide_idx = archive.sample_guide_index(rng)
        if guide_idx is None:
            G = P[rng.integers(0, n_particles)]
        else:
            G = archive.X[guide_idx]

        r1 = rng.random(size=(n_particles, n_bits))
        r2 = rng.random(size=(n_particles, n_bits))

        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (G - X)
        V = np.clip(V, -vmax, vmax)

        prob = sigmoid(V)
        X = (rng.random(size=(n_particles, n_bits)) < prob).astype(int)

        for i in range(n_particles):
            X[i] = repair_w1_capacity(X[i], w1_arr, capacity_w1, rng)

        total_value, total_w1, total_w2, feasible, F = evaluate_population(X, value, w1_arr, w2_arr, capacity_w1)

        update_personal_best(P, PF, X, F, rng)

        archive.add(X, F)

        best_value_so_far = max(best_value_so_far, float(np.max(total_value)))
        history.append(best_value_so_far)

    runtime = time.time() - start

    # Convert archive objectives back to readable metrics
    # f1 is -value
    pareto_values = (-archive.F[:, 0]).astype(float)
    pareto_w2 = (archive.F[:, 1]).astype(float)

    return {
        "archive_X": archive.X,
        "pareto_value": pareto_values,
        "pareto_w2": pareto_w2,
        "history_best_value": history,
        "runtime_s": float(runtime),
        "capacity_w1": int(capacity_w1),
    }
