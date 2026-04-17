
from __future__ import annotations
import math
from typing import Tuple, Optional
import numpy as np

try:
    from .mcts_core import QEdgeStats, NodeStats, ScalarQ
except ImportError:
    from mcts_core import QEdgeStats, NodeStats, ScalarQ


class BaseSelector:
    def __init__(self, k: int, C: float = 1.5, N: int = 100, K: int = 200, p: float = 1.0, ucb_c: float = 1.0):
        self.k = k
        self.C = C
        self.N = N
        self.K = K
        self.p = p
        self.ucb_c = ucb_c

    # hook to initialize edge distribution
    def prepare_edge(self, qe: QEdgeStats, reward_range: Tuple[float, float]):
        pass

    def select_action(self, node: NodeStats, rng: np.random.Generator) -> int:
        raise NotImplementedError

    # compute V backup power mean after updating edges
    @staticmethod
    def compute_v_backup(node: NodeStats, p: float):
        if node.visits == 0:
            node.v_value = 0.0
            return
        values = []
        weights = []
        for a, qe in node.edges.items():
            values.append(qe.q_expected())
            w = qe.visits / max(1, node.visits)
            weights.append(w)
        # power mean
        if p == float('inf'):
            node.v_value = max(values) if values else 0.0
        else:
            acc = 0.0
            for v, w in zip(values, weights):
                acc += (w * (v ** p))
            node.v_value = (acc ** (1.0 / p)) if values else 0.0

# --------- Scalar TS + Optimism (ablation) ---------
class ScalarTSOptSelector(BaseSelector):
    def prepare_edge(self, qe: QEdgeStats, reward_range: Tuple[float, float]):
        if qe.scalar is None:
            qe.scalar = ScalarQ()
        self.reward_range = reward_range

    def select_action(self, node: NodeStats, rng: np.random.Generator) -> int:
        for a, qe in node.edges.items():
            if qe.visits == 0:
                return a
        best_a = 0
        best_score = -1e18
        for a, qe in node.edges.items():
            ts = qe.scalar.sample_ts(rng, self.reward_range) if qe.scalar is not None else 0.0
            bonus = self.C * (node.visits ** 0.25) / (max(1, qe.visits) ** 0.5)
            score = ts + bonus
            if score > best_score:
                best_score = score
                best_a = a
        return best_a


# --------- UCT (UCB1) ---------
class UCTSelector(BaseSelector):
    def __init__(self, k: int, ucb_c: float = 1.0):
        super().__init__(k=k, ucb_c=ucb_c)

    def prepare_edge(self, qe: QEdgeStats, reward_range: Tuple[float, float]):
        if qe.scalar is None:
            qe.scalar = ScalarQ()

    def select_action(self, node: NodeStats, rng: np.random.Generator) -> int:
        for a, qe in node.edges.items():
            if qe.visits == 0:
                return a
        best_a = 0
        best = -1e18
        for a, qe in node.edges.items():
            q = qe.scalar.expected()
            bonus = self.ucb_c * math.sqrt(math.log(max(1, node.visits)) / max(1, qe.visits))
            s = q + bonus
            if s > best:
                best = s
                best_a = a
        return best_a


# --------- Power-UCT (UCB with power-mean V-backup) ---------
class PowerUCTSelector(UCTSelector):
    # identical to UCT for action selection; difference is in V-backup which the runner controls via 'p'
    pass
