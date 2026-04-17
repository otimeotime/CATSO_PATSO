from __future__ import annotations
import numpy as np

class CATSO:
    """Categorical TS + polynomial optimism using sampled-distribution CVaR.

    Public API matches the existing selector interface:
      - prepare_edge(qe, reward_range)
      - select_action(node, rng)
      - compute_v_backup(node, p)

    Notes:
      * V-node backups stay scalar/power-mean, as in the paper.
      * Action selection uses CVaR of the sampled categorical law, not its mean.
      * `tau=1.0` recovers the ordinary mean.
    """

    class _CategoricalQ:
        def __init__(self, n_atoms: int, qmin: float = 0.0, qmax: float = 1e-3):
            assert n_atoms >= 2
            self.n_atoms = int(n_atoms)
            self.qmin = float(qmin)
            self.qmax = float(max(qmax, qmin + 1e-3))
            self.alpha = np.ones(self.n_atoms, dtype=float)
            self.atoms = np.linspace(self.qmin, self.qmax, self.n_atoms)

        def _empirical_mass(self) -> np.ndarray:
            return np.maximum(self.alpha - 1.0, 0.0)

        def expected(self) -> float:
            mass = self._empirical_mass()
            total = float(np.sum(mass))
            if total <= 0.0:
                return 0.0
            return float(np.dot(self.atoms, mass / total))

        @staticmethod
        def _project_point(z: float, atoms: np.ndarray) -> np.ndarray:
            z = float(z)
            atoms = np.asarray(atoms, dtype=float)
            out = np.zeros_like(atoms, dtype=float)
            if z <= atoms[0]:
                out[0] = 1.0
                return out
            if z >= atoms[-1]:
                out[-1] = 1.0
                return out

            dz = float(atoms[1] - atoms[0])
            b = (z - float(atoms[0])) / dz
            left = int(np.floor(b))
            left = max(0, min(left, len(atoms) - 2))
            frac = b - left
            out[left] = 1.0 - frac
            out[left + 1] = frac
            return out

        @staticmethod
        def _disc_cvar(values: np.ndarray, weights: np.ndarray, tau: float) -> float:
            tau = float(tau)
            if tau <= 0.0 or tau > 1.0:
                raise ValueError("tau must be in (0, 1].")
            total = float(np.sum(weights))
            if total <= 0.0:
                return float(np.mean(values))
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
            L = rng.dirichlet(self.alpha)
            return self._disc_cvar(self.atoms, L, tau)

        def _regrid(self, new_min: float, new_max: float):
            new_atoms = np.linspace(new_min, new_max, self.n_atoms)
            new_alpha = np.ones(self.n_atoms, dtype=float)
            for old_atom, old_mass in zip(self.atoms, self._empirical_mass()):
                if old_mass <= 0.0:
                    continue
                new_alpha += old_mass * self._project_point(old_atom, new_atoms)
            self.qmin = float(new_min)
            self.qmax = float(new_max)
            self.atoms = new_atoms
            self.alpha = new_alpha

        def update(self, q_sample: float):
            q_sample = float(q_sample)
            if q_sample < self.qmin or q_sample > self.qmax:
                new_min = min(self.qmin, q_sample)
                new_max = max(self.qmax, q_sample)
                if new_max - new_min < 1e-6:
                    new_max = new_min + 1e-3
                self._regrid(new_min, new_max)
            self.alpha += self._project_point(q_sample, self.atoms)

    def __init__(self, k: int, C: float = 1.5, N: int = 100, p: float = 1.0, tau: float = 1.0):
        self.k = int(k)
        self.C = float(C)
        self.N = int(N)
        self.p = p
        self.tau = float(tau)
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be in (0, 1].")

    def prepare_edge(self, qe, reward_range):
        if getattr(qe, "cat", None) is None:
            qe.cat = self._CategoricalQ(self.N)

    def select_action(self, node, rng: np.random.Generator) -> int:
        unvisited = [a for a, qe in node.edges.items() if qe.visits == 0]
        if unvisited:
            return int(rng.choice(np.asarray(unvisited, dtype=int)))

        best_actions = []
        best_score = -float("inf")
        for a, qe in node.edges.items():
            cvar = qe.cat.sample_cvar(rng, self.tau) if qe.cat is not None else 0.0
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
