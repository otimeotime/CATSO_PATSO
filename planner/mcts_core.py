
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Callable

import numpy as np


# ----------------- Power mean helper -----------------
def power_mean(values: List[float], weights: List[float], p: float) -> float:
    assert len(values) == len(weights) and len(values) > 0
    wsum = sum(weights)
    if wsum <= 0:
        return float(np.mean(values))
    if p == float('inf'):
        return max(values)
    # numeric stability for p close to 0 not needed (we only use p>=1)
    acc = 0.0
    for v, w in zip(values, weights):
        acc += (w / wsum) * (v ** p)
    return acc ** (1.0 / p)


# ----------------- Distributions per Q-edge -----------------
class CategoricalQ:
    def __init__(self, N: int, qmin: float, qmax: float):
        assert N >= 2
        self.N = N
        self.qmin = qmin
        self.qmax = max(qmax, qmin + 1e-3)
        # dirichlet counts (alphas), initialized to 1
        self.alpha = np.ones(N, dtype=float)
        self.total = float(N)  # sum(alpha)
        self.atoms = np.linspace(self.qmin, self.qmax, N)

    def expected(self) -> float:
        probs = self.alpha / np.sum(self.alpha)
        return float(np.dot(self.atoms, probs))

    def sample_ts(self, rng: np.random.Generator) -> float:
        L = rng.dirichlet(self.alpha)
        return float(np.dot(self.atoms, L))

    def _regrid(self, new_min: float, new_max: float):
        # re-project old counts to closest bins on new grid
        new_atoms = np.linspace(new_min, new_max, self.N)
        new_alpha = np.ones(self.N, dtype=float)  # keep symmetric prior 1
        for a_old, w in zip(self.atoms, self.alpha):
            j = int(np.argmin(np.abs(new_atoms - a_old)))
            new_alpha[j] += w - 1.0  # subtract the prior that we added
        self.qmin, self.qmax = float(new_min), float(new_max)
        self.atoms = new_atoms
        self.alpha = new_alpha

    def update(self, q_sample: float):
        # expand support if needed, then increment nearest bin
        if q_sample < self.qmin or q_sample > self.qmax:
            new_min = min(self.qmin, q_sample)
            new_max = max(self.qmax, q_sample)
            # keep a minimum span to avoid singular grid
            if new_max - new_min < 1e-6:
                new_max = new_min + 1e-3
            self._regrid(new_min, new_max)
        j = int(np.argmin(np.abs(self.atoms - q_sample))) 
        self.alpha[j] += 1.0
        self.total += 1.0


class ParticleQ:
    def __init__(self, K: int, tol: float = 1e-9):
        assert K >= 1
        self.K = K
        self.tol = tol
        # keep sorted particles and integer weights
        self.values: List[float] = []
        self.weights: List[int] = []

    def _insert_sorted(self, z: float, w: int = 1):
        # binary search insert into sorted values
        import bisect
        i = bisect.bisect_left(self.values, z)
        self.values.insert(i, z)
        self.weights.insert(i, w)

    def expected(self) -> float:
        if not self.values:
            return 0.0
        wsum = float(sum(self.weights))
        return float(sum(v * w for v, w in zip(self.values, self.weights)) / wsum)

    def sample_ts(self, rng: np.random.Generator) -> float:
        if not self.values:
            return 0.0
        # Dirichlet over integer counts
        alpha = np.array(self.weights, dtype=float)
        L = rng.dirichlet(alpha)
        return float(np.dot(np.array(self.values), L))

    def _merge_closest_pair(self):
        # merge two adjacent particles with smallest gap to free one slot
        if len(self.values) < 2:
            return
        gaps = [abs(self.values[i+1] - self.values[i]) for i in range(len(self.values)-1)]
        j = int(np.argmin(gaps))
        v1, v2 = self.values[j], self.values[j+1]
        w1, w2 = self.weights[j], self.weights[j+1]
        v_new = (w1 * v1 + w2 * v2) / (w1 + w2)
        w_new = w1 + w2
        # replace indices j and j+1 by new
        self.values[j:j+2] = [v_new]
        self.weights[j:j+2] = [w_new]

    def update(self, q_sample: float):
        # if within tol of an existing particle, bump weight; else insert
        for i, v in enumerate(self.values):
            if abs(v - q_sample) <= self.tol:
                self.weights[i] += 1
                break
        else:
            # need to insert
            if len(self.values) < self.K:
                self._insert_sorted(q_sample, 1)
            else:
                # merge best pair, then insert
                self._merge_closest_pair()
                self._insert_sorted(q_sample, 1)


class ScalarQ:
    """Scalar-mean baseline for TS+optimism ablation."""
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def expected(self) -> float:
        return self.sum / self.n if self.n > 0 else 0.0

    def sample_ts(self, rng: np.random.Generator, reward_range: Tuple[float, float]) -> float:
        # simple Gaussian TS around the empirical mean; variance shrinks as 1/(n+1)
        mu = self.expected()
        lo, hi = reward_range
        # scale with range
        rng_scale = (hi - lo)
        std = rng_scale / math.sqrt(max(1, self.n + 1))
        return float(rng.normal(mu, std))

    def update(self, q_sample: float):
        self.n += 1
        self.sum += q_sample
        self.sumsq += q_sample * q_sample


# ----------------- Node & Edge statistics -----------------
@dataclass
class QEdgeStats:
    visits: int = 0
    q_mean: Optional[float] = None
    # Only one of the following distribution holders is used depending on algorithm:
    cat: Optional[CategoricalQ] = None
    part: Optional[ParticleQ] = None
    scalar: Optional[ScalarQ] = None

    def q_expected(self) -> float:
        if self.q_mean is not None:
            return self.q_mean
        if self.cat is not None:
            return self.cat.expected()
        if self.part is not None:
            return self.part.expected()
        if self.scalar is not None:
            return self.scalar.expected()
        return 0.0


@dataclass
class NodeStats:
    visits: int = 0
    # action -> QEdgeStats
    edges: Dict[int, QEdgeStats] = field(default_factory=dict)
    v_value: float = 0.0  # backup value at V-node (power mean)

    def ensure_edge(self, a: int) -> QEdgeStats:
        if a not in self.edges:
            self.edges[a] = QEdgeStats()
        return self.edges[a]
