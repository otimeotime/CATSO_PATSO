from __future__ import annotations

import bisect
import numpy as np


class PATSO:
    """Particle TS + polynomial optimism using sampled-distribution CVaR.

    Public API matches the existing selector interface:
      - prepare_edge(qe, reward_range)
      - select_action(node, rng)
      - compute_v_backup(node, p)

    Notes:
      * V-node backups stay scalar/power-mean, as in the paper.
      * Action selection uses CVaR of the sampled particle law, not its mean.
      * `tau=1.0` recovers the ordinary mean.
    """

    class _ParticleQ:
        def __init__(self, K: int, tol: float = 1e-9):
            assert K >= 1
            self.K = int(K)
            self.tol = float(tol)
            self.values = []
            self.weights = []

        def expected(self) -> float:
            if not self.values:
                return 0.0
            total = float(sum(self.weights))
            return float(sum(v * w for v, w in zip(self.values, self.weights)) / total)

        @staticmethod
        def _disc_cvar(values: np.ndarray, weights: np.ndarray, tau: float) -> float:
            tau = float(tau)
            if tau <= 0.0 or tau > 1.0:
                raise ValueError("tau must be in (0, 1].")
            total = float(np.sum(weights))
            if total <= 0.0:
                return float(np.mean(values)) if len(values) else 0.0
            w = np.asarray(weights, dtype=float) / total
            x = np.asarray(values, dtype=float)
            order = np.argsort(x)
            x = x[order]
            w = w[order]
            csum = np.cumsum(w)
            if tau >= 1.0 - 1e-12:
                return float(np.dot(x, w))
            k = int(np.searchsorted(csum, tau, side="left"))
            k = min(k, len(x) - 1)
            prev = float(csum[k - 1]) if k > 0 else 0.0
            tail_sum = float(np.dot(w[:k], x[:k])) if k > 0 else 0.0
            tail_sum += (tau - prev) * float(x[k])
            return tail_sum / tau

        def sample_cvar(self, rng: np.random.Generator, tau: float) -> float:
            if not self.values:
                return 0.0
            L = rng.dirichlet(np.asarray(self.weights, dtype=float))
            return self._disc_cvar(np.asarray(self.values, dtype=float), L, tau)

        def _insert_sorted(self, z: float, w: int = 1):
            i = bisect.bisect_left(self.values, z)
            self.values.insert(i, float(z))
            self.weights.insert(i, int(w))

        def _merge_closest_pair(self):
            if len(self.values) < 2:
                return
            gaps = [abs(self.values[i + 1] - self.values[i]) for i in range(len(self.values) - 1)]
            j = int(np.argmin(gaps))
            v1, v2 = self.values[j], self.values[j + 1]
            w1, w2 = self.weights[j], self.weights[j + 1]
            v_new = (w1 * v1 + w2 * v2) / (w1 + w2)
            self.values[j:j + 2] = [float(v_new)]
            self.weights[j:j + 2] = [int(w1 + w2)]

        def update(self, q_sample: float):
            q_sample = float(q_sample)
            for i, v in enumerate(self.values):
                if abs(v - q_sample) <= self.tol:
                    self.weights[i] += 1
                    return
            if len(self.values) >= self.K:
                self._merge_closest_pair()
            self._insert_sorted(q_sample, 1)

    def __init__(self, k: int, C: float = 1.5, K: int = 200, p: float = 1.0, tau: float = 1.0):
        self.k = int(k)
        self.C = float(C)
        self.K = int(K)
        self.p = p
        self.tau = float(tau)
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be in (0, 1].")

    def prepare_edge(self, qe, reward_range):
        if getattr(qe, "part", None) is None:
            qe.part = self._ParticleQ(self.K)

    def select_action(self, node, rng: np.random.Generator) -> int:
        unvisited = [a for a, qe in node.edges.items() if qe.visits == 0]
        if unvisited:
            return int(rng.choice(np.asarray(unvisited, dtype=int)))

        best_actions = []
        best_score = -float("inf")
        for a, qe in node.edges.items():
            cvar = qe.part.sample_cvar(rng, self.tau) if qe.part is not None else 0.0
            bonus = self.C * (max(1, node.visits) ** 0.25) / (max(1, qe.visits) ** 0.5)
            score = cvar + bonus
            if score > best_score:
                best_score = score
                best_actions = [a]
            elif score >= best_score - 1e-12:
                best_actions.append(a)
        return int(rng.choice(np.asarray(best_actions, dtype=int)))

    @staticmethod
    def compute_v_backup(node, p: float):
        if node.visits == 0:
            node.v_value = 0.0
            return
        values, weights = [], []
        for qe in node.edges.values():
            values.append(qe.q_expected())
            weights.append(qe.visits / max(1, node.visits))
        if not values:
            node.v_value = 0.0
        elif p == float("inf"):
            node.v_value = max(values)
        else:
            acc = 0.0
            for v, w in zip(values, weights):
                acc += w * (v ** p)
            node.v_value = float(acc ** (1.0 / p))
