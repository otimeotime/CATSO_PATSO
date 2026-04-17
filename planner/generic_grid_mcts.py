from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, Sequence, Tuple

import numpy as np

try:
    from .mcts_core import NodeStats
except ImportError:
    from mcts_core import NodeStats


Observation = Hashable
StateKey = Tuple[Observation, int]


def discrete_cvar(values, weights, alpha: float) -> float:
    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    if len(values) == 0:
        return 0.0
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0.0:
        return float(np.mean(x))
    w = w / total
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    csum = np.cumsum(w)
    k = int(np.searchsorted(csum, alpha, side="left"))
    prev = float(csum[k - 1]) if k > 0 else 0.0
    tail_sum = float(np.dot(w[:k], x[:k])) if k > 0 else 0.0
    tail_sum += (alpha - prev) * float(x[k])
    return tail_sum / alpha


def empirical_qedge_cvar(qe, alpha: float) -> float:
    if getattr(qe, "cat", None) is not None:
        values = np.asarray(qe.cat.atoms, dtype=float)
        weights = np.maximum(np.asarray(qe.cat.alpha, dtype=float) - 1.0, 0.0)
        return discrete_cvar(values, weights, alpha)
    if getattr(qe, "part", None) is not None:
        return discrete_cvar(qe.part.values, qe.part.weights, alpha)
    return qe.q_expected()


class GenericGridMCTS:
    def __init__(
        self,
        env_factory: Callable[[int | None], Any],
        root_observation: Observation,
        selector,
        gamma: float,
        reward_range: Tuple[float, float],
        backup_p: float,
        recommendation_mode: str,
        planner_alpha: float,
        seed: int,
    ):
        self.env_factory = env_factory
        self.selector = selector
        self.gamma = float(gamma)
        self.reward_range = reward_range
        self.backup_p = backup_p
        self.recommendation_mode = recommendation_mode
        self.planner_alpha = float(planner_alpha)
        self.rng = np.random.default_rng(seed)
        self.nodes: Dict[StateKey, NodeStats] = {}
        self.root_key: StateKey = (root_observation, 0)
        self._is_distributional = selector.__class__.__name__.lower() in {"catso", "patso"}

    def run_until(self, n_sims: int) -> None:
        current = self.nodes.get(self.root_key, NodeStats()).visits if self.root_key in self.nodes else 0
        for _ in range(current, n_sims):
            self._simulate_once()

    def _simulate_once(self) -> None:
        sim_seed = int(self.rng.integers(0, 2**31 - 1))
        env = self.env_factory(sim_seed)
        obs, _ = env.reset(seed=sim_seed)
        self._simulate_v(env, (obs, 0))

    def _ensure_node(self, state_key: StateKey, legal_actions: Sequence[int]) -> NodeStats:
        node = self.nodes.setdefault(state_key, NodeStats())
        for action in legal_actions:
            qe = node.ensure_edge(action)
            self.selector.prepare_edge(qe, self.reward_range)
        return node

    def _update_edge(self, node: NodeStats, action: int, q_sample: float) -> None:
        qe = node.edges[action]
        qe.visits += 1
        if qe.q_mean is None:
            qe.q_mean = q_sample
        else:
            qe.q_mean += (q_sample - qe.q_mean) / qe.visits
        if qe.cat is not None:
            qe.cat.update(q_sample)
        if qe.part is not None:
            qe.part.update(q_sample)
        if qe.scalar is not None:
            qe.scalar.update(q_sample)
        node.visits += 1
        self.selector.compute_v_backup(node, p=self.backup_p)

    def _rollout(self, env: Any) -> float:
        total = 0.0
        discount = 1.0
        while True:
            actions = np.asarray(env.legal_actions(), dtype=int)
            action = int(self.rng.choice(actions))
            _, reward, terminated, truncated, _ = env.step(action)
            total += discount * float(reward)
            if terminated or truncated:
                return total
            discount *= self.gamma

    def _simulate_v(self, env: Any, state_key: StateKey) -> float:
        node = self._ensure_node(state_key, env.legal_actions())
        action = self.selector.select_action(node, self.rng)
        next_obs, reward, terminated, truncated, info = env.step(action)
        reward = float(reward)

        if terminated or truncated:
            q_sample = reward
        else:
            child_key: StateKey = (next_obs, int(info.get("steps", 0)))
            child_node = self.nodes.get(child_key)
            if child_node is None:
                child_value = self._rollout(env)
                child_node = self.nodes.setdefault(child_key, NodeStats())
                child_node.v_value = child_value
            else:
                child_value = self._simulate_v(env, child_key)
            q_sample = reward + self.gamma * child_value

        self._update_edge(node, action, q_sample)
        return node.v_value

    def recommend_root_action(self) -> int:
        root = self.nodes.get(self.root_key)
        if root is None or not root.edges:
            return 0
        best_action = 0
        best_score = -float("inf")
        for action, qe in root.edges.items():
            if self.recommendation_mode == "mean" or not self._is_distributional or self.planner_alpha >= 1.0 - 1e-12:
                score = qe.q_expected()
            else:
                score = empirical_qedge_cvar(qe, self.planner_alpha)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
