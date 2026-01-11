from __future__ import annotations

import numpy as np

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Minimization form for both objectives.
    We convert value to negative value so both are minimization.
    a dominates b if a is no worse in all objectives and better in at least one.
    """
    return np.all(a <= b) and np.any(a < b)

def fast_nondominated_filter(F: np.ndarray):
    """
    Returns indices of nondominated points in F.
    F shape is (k, 2) for two objectives.
    """
    k = F.shape[0]
    keep = np.ones(k, dtype=bool)
    for i in range(k):
        if not keep[i]:
            continue
        for j in range(k):
            if i == j or not keep[j]:
                continue
            if dominates(F[j], F[i]):
                keep[i] = False
                break
    return np.where(keep)[0]

def crowding_distance(F: np.ndarray) -> np.ndarray:
    """
    Crowding distance for 2 objectives.
    Larger is more diverse.
    """
    n = F.shape[0]
    if n == 0:
        return np.array([])
    if n <= 2:
        return np.full(n, np.inf)

    dist = np.zeros(n, dtype=float)
    for m in range(F.shape[1]):
        order = np.argsort(F[:, m])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        fmin = F[order[0], m]
        fmax = F[order[-1], m]
        if fmax == fmin:
            continue
        for i in range(1, n - 1):
            prev_v = F[order[i - 1], m]
            next_v = F[order[i + 1], m]
            dist[order[i]] += (next_v - prev_v) / (fmax - fmin)
    return dist

class ParetoArchive:
    def __init__(self, max_size: int = 200):
        self.max_size = int(max_size)
        self.X = np.empty((0, 0), dtype=int)
        self.F = np.empty((0, 2), dtype=float)

    def add(self, X_new: np.ndarray, F_new: np.ndarray):
        """
        X_new shape (p, n_bits)
        F_new shape (p, 2)
        """
        if X_new.size == 0:
            return

        if self.X.size == 0:
            self.X = X_new.copy()
            self.F = F_new.copy()
        else:
            self.X = np.vstack([self.X, X_new])
            self.F = np.vstack([self.F, F_new])

        nd_idx = fast_nondominated_filter(self.F)
        self.X = self.X[nd_idx]
        self.F = self.F[nd_idx]

        if self.F.shape[0] > self.max_size:
            cd = crowding_distance(self.F)
            keep = np.argsort(-cd)[: self.max_size]
            self.X = self.X[keep]
            self.F = self.F[keep]

    def sample_guide_index(self, rng: np.random.Generator):
        """
        Choose a global guide from archive using crowding-based probabilities.
        """
        if self.F.shape[0] == 0:
            return None
        if self.F.shape[0] == 1:
            return 0

        cd = crowding_distance(self.F)
        finite = np.isfinite(cd)
        if np.any(finite):
            w = cd.copy()
            w[~finite] = np.max(w[finite])
            w = w + 1e-12
            p = w / np.sum(w)
            return int(rng.choice(np.arange(len(p)), p=p))
        return int(rng.integers(0, self.F.shape[0]))
