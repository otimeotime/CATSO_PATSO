from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Make sibling files importable when this script is run from anywhere.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from oracle_evaluable_MDP import RiskyShortcutGridworld, ThinIceFrozenLakePlus, WindSpec  # type: ignore
from planner.CATSO import CATSO  # type: ignore
from planner.PATSO import PATSO  # type: ignore
from planner.baselines import ScalarTSOptSelector, UCTSelector, PowerUCTSelector  # type: ignore
from planner.generic_grid_mcts import GenericGridMCTS  # type: ignore


Coord = Tuple[int, int]


@dataclass(frozen=True)
class EnvConfig:
    env_name: str = "corridor"
    N: int = 10
    slip_prob: float = 0.10
    wind_prob: float = 0.20
    thin_frac: float = 0.15
    break_prob: float = 0.10
    step_cost: float = -1.0
    goal_reward: float = 50.0
    cliff_penalty: float = -100.0
    break_penalty: float = -120.0
    max_steps: int = 100
    observation_mode: str = "tuple"


@dataclass(frozen=True)
class PlannerConfig:
    gamma: float = 0.99
    planner_alpha: float = 1.0
    eval_cvar_alpha: float = 0.10
    recommendation_mode: str = "planner"  # planner | mean
    methods: Tuple[str, ...] = ("catso", "patso", "uct", "poweruct", "scalarts")
    budgets: Tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
    seeds: int = 100
    catso_N: int = 100
    patso_K: int = 200
    bonus_C: float = 8.0
    dist_backup_p: float = math.inf
    poweruct_p: float = 2.0
    uct_c: float | None = None
    scalar_C: float = 8.0
    oracle_rollouts: int = 5000
    output_dir: str = "./eval_outputs"


def parse_float_or_inf(x: str) -> float:
    if x.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    return float(x)


def parse_int_list(text: str | None) -> Tuple[int, ...]:
    if text is None or text.strip() == "":
        return tuple()
    vals = sorted({int(v.strip()) for v in text.split(",") if v.strip()})
    return tuple(v for v in vals if v >= 1)


def env_label(cfg: EnvConfig) -> str:
    return "catastrophe_corridor" if cfg.env_name == "corridor" else "thin_ice_frozen_lake_plus"


def env_start(cfg: EnvConfig) -> Coord:
    if cfg.env_name == "corridor":
        return (cfg.N - 1, 0)
    if cfg.env_name == "thinice":
        return (0, 0)
    raise ValueError(f"Unknown environment: {cfg.env_name}")


def env_start_observation(cfg: EnvConfig):
    start = env_start(cfg)
    if cfg.observation_mode == "tuple":
        return start
    return start[0] * cfg.N + start[1]


def make_env(cfg: EnvConfig, seed: int | None = None) -> RiskyShortcutGridworld | ThinIceFrozenLakePlus:
    if cfg.env_name == "corridor":
        return RiskyShortcutGridworld(
            N=cfg.N,
            slip_prob=cfg.slip_prob,
            wind=WindSpec(wind_prob=cfg.wind_prob),
            step_cost=cfg.step_cost,
            goal_reward=cfg.goal_reward,
            cliff_penalty=cfg.cliff_penalty,
            observation_mode=cfg.observation_mode,
            max_steps=cfg.max_steps,
            seed=seed,
        )
    if cfg.env_name == "thinice":
        return ThinIceFrozenLakePlus(
            N=cfg.N,
            slip_prob=cfg.slip_prob,
            thin_frac=cfg.thin_frac,
            break_prob=cfg.break_prob,
            step_cost=cfg.step_cost,
            goal_reward=cfg.goal_reward,
            break_penalty=cfg.break_penalty,
            observation_mode=cfg.observation_mode,
            max_steps=cfg.max_steps,
            seed=seed,
        )
    raise ValueError(f"Unknown environment: {cfg.env_name}")


class CorridorOracle:
    """Exact expected-value DP oracle plus exact root backup-law CVaR oracle.

    The expected-value oracle matches the tabular DP setup described in the paper.
    The CVaR oracle is computed on the exact *root backup law*
        D(s0,a) = Law( r(s0,a,S') + gamma * V*(S') ),
    which is discrete and inexpensive to evaluate exactly for this environment.
    """

    ACTIONS = (0, 1, 2, 3)
    DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def __init__(self, cfg: EnvConfig, gamma: float, cvar_alpha: float):
        self.cfg = cfg
        self.gamma = float(gamma)
        self.cvar_alpha = float(cvar_alpha)
        self.start: Coord = (cfg.N - 1, 0)
        self.goal: Coord = (cfg.N - 1, cfg.N - 1)
        self.cliff_cells = {(cfg.N - 1, c) for c in range(1, cfg.N - 1)}
        self.windy_cols = list(range(1, cfg.N - 1))
        self.positions = [(r, c) for r in range(cfg.N) for c in range(cfg.N)]
        self.V: List[Dict[Coord, float]] = [dict() for _ in range(cfg.max_steps + 1)]
        self.Q_root: Dict[int, float] = {}
        self.root_backup_law: Dict[int, List[Tuple[float, float]]] = {}
        self.root_cvar: Dict[int, float] = {}
        self._solve()

    def _clip(self, r: int, c: int) -> Coord:
        return (max(0, min(self.cfg.N - 1, r)), max(0, min(self.cfg.N - 1, c)))

    def _is_terminal_cell(self, pos: Coord) -> bool:
        return pos == self.goal or pos in self.cliff_cells

    def _action_exec_probs(self, chosen_action: int) -> Dict[int, float]:
        base = self.cfg.slip_prob / 4.0
        probs = {a: base for a in self.ACTIONS}
        probs[chosen_action] += 1.0 - self.cfg.slip_prob
        return probs

    def transition_distribution(self, pos: Coord, action: int) -> List[Tuple[float, Coord, float, bool]]:
        """Return aggregated outcomes (prob, next_pos, reward, terminated_before_horizon)."""
        if self._is_terminal_cell(pos):
            return [(1.0, pos, 0.0, True)]

        outcomes: Dict[Tuple[Coord, float, bool], float] = {}
        for exec_action, p_exec in self._action_exec_probs(action).items():
            if p_exec <= 0:
                continue
            dr, dc = self.DELTAS[exec_action]
            r0, c0 = pos
            nr, nc = self._clip(r0 + dr, c0 + dc)

            wind_branches: List[Tuple[float, Coord]] = []
            if nc in self.windy_cols:
                wind_branches.append((1.0 - self.cfg.wind_prob, (nr, nc)))
                wind_branches.append((self.cfg.wind_prob, self._clip(nr + 1, nc)))
            else:
                wind_branches.append((1.0, (nr, nc)))

            for p_wind, final_pos in wind_branches:
                p = p_exec * p_wind
                if p <= 0:
                    continue
                reward = self.cfg.step_cost
                terminated = False
                if final_pos in self.cliff_cells:
                    reward = self.cfg.cliff_penalty
                    terminated = True
                elif final_pos == self.goal:
                    reward = self.cfg.step_cost + self.cfg.goal_reward
                    terminated = True
                key = (final_pos, float(reward), terminated)
                outcomes[key] = outcomes.get(key, 0.0) + p

        return [(p, next_pos, reward, done) for (next_pos, reward, done), p in outcomes.items()]

    def _solve(self) -> None:
        # terminal value at the time limit is zero
        for pos in self.positions:
            self.V[self.cfg.max_steps][pos] = 0.0

        for t in range(self.cfg.max_steps - 1, -1, -1):
            for pos in self.positions:
                if self._is_terminal_cell(pos):
                    self.V[t][pos] = 0.0
                    continue
                best_q = -float("inf")
                for a in self.ACTIONS:
                    q = 0.0
                    for p, next_pos, reward, done in self.transition_distribution(pos, a):
                        if done or (t + 1 >= self.cfg.max_steps):
                            q += p * reward
                        else:
                            q += p * (reward + self.gamma * self.V[t + 1][next_pos])
                    best_q = max(best_q, q)
                self.V[t][pos] = best_q

        # Root expected-action values and root backup law.
        for a in self.ACTIONS:
            q = 0.0
            support: Dict[float, float] = {}
            for p, next_pos, reward, done in self.transition_distribution(self.start, a):
                z = reward if (done or 1 >= self.cfg.max_steps) else (reward + self.gamma * self.V[1][next_pos])
                q += p * z
                support[z] = support.get(z, 0.0) + p
            law = sorted(support.items(), key=lambda kv: kv[0])
            self.Q_root[a] = q
            self.root_backup_law[a] = law
            self.root_cvar[a] = discrete_cvar([x for x, _ in law], [w for _, w in law], self.cvar_alpha)

        self.best_q = max(self.Q_root.values())
        self.optimal_actions = {a for a, q in self.Q_root.items() if abs(q - self.best_q) <= 1e-12}
        self.best_root_cvar = max(self.root_cvar.values())
        self.optimal_cvar_actions = {a for a, v in self.root_cvar.items() if abs(v - self.best_root_cvar) <= 1e-12}


class ThinIceRolloutOracle:
    """Monte Carlo root-action oracle for ThinIceFrozenLakePlus.

    ThinIceFrozenLakePlus samples a hidden thin-ice map each episode, so the exact
    tabular root oracle used for RiskyShortcutGridworld does not apply under the
    current position-only state abstraction. We therefore estimate root-action
    values and CVaR via many start-state Monte Carlo rollouts with a random
    continuation policy.
    """

    ACTIONS = (0, 1, 2, 3)

    def __init__(self, cfg: EnvConfig, gamma: float, cvar_alpha: float, num_rollouts: int, seed: int = 0):
        self.cfg = cfg
        self.gamma = float(gamma)
        self.cvar_alpha = float(cvar_alpha)
        self.num_rollouts = int(num_rollouts)
        self.seed = int(seed)
        self.start: Coord = env_start(cfg)
        self.Q_root: Dict[int, float] = {}
        self.root_backup_law: Dict[int, List[Tuple[float, float]]] = {}
        self.root_cvar: Dict[int, float] = {}
        self._solve()

    def _simulate_random_return(self, chosen_action: int, episode_seed: int) -> float:
        env = make_env(self.cfg, seed=episode_seed)
        _, _ = env.reset(seed=episode_seed)
        _, reward, terminated, truncated, _ = env.step(chosen_action)
        total = float(reward)
        discount = self.gamma
        rng = np.random.default_rng(episode_seed + 1)
        while not (terminated or truncated):
            actions = np.asarray(env.legal_actions(), dtype=int)
            action = int(rng.choice(actions))
            _, reward, terminated, truncated, _ = env.step(action)
            total += discount * float(reward)
            discount *= self.gamma
        return total

    def _solve(self) -> None:
        for action in self.ACTIONS:
            returns = np.asarray(
                [self._simulate_random_return(action, self.seed + 1000003 * action + i) for i in range(self.num_rollouts)],
                dtype=float,
            )
            self.Q_root[action] = float(np.mean(returns))
            values, counts = np.unique(returns, return_counts=True)
            law = [(float(v), float(c) / self.num_rollouts) for v, c in zip(values, counts)]
            self.root_backup_law[action] = law
            self.root_cvar[action] = discrete_cvar(values, counts, self.cvar_alpha)

        self.best_q = max(self.Q_root.values())
        self.optimal_actions = {a for a, q in self.Q_root.items() if abs(q - self.best_q) <= 1e-12}
        self.best_root_cvar = max(self.root_cvar.values())
        self.optimal_cvar_actions = {a for a, v in self.root_cvar.items() if abs(v - self.best_root_cvar) <= 1e-12}


def discounted_return_bounds(cfg: EnvConfig, gamma: float) -> Tuple[float, float]:
    terminal_penalty = cfg.cliff_penalty if cfg.env_name == "corridor" else cfg.break_penalty
    rmin = min(cfg.step_cost, cfg.step_cost + cfg.goal_reward, terminal_penalty)
    rmax = max(cfg.step_cost, cfg.step_cost + cfg.goal_reward, terminal_penalty)
    if abs(gamma - 1.0) < 1e-12:
        return cfg.max_steps * rmin, cfg.max_steps * rmax
    factor = (1.0 - gamma ** cfg.max_steps) / (1.0 - gamma)
    return factor * rmin, factor * rmax


def discrete_cvar(values: Sequence[float], weights: Sequence[float], alpha: float) -> float:
    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    if len(values) == 0:
        return 0.0
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0:
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


def mean_and_ci(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = arr.mean(axis=0)
    if arr.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    ci = 1.96 * arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0])
    return mean, ci


def make_selector(method: str, env_cfg: EnvConfig, cfg: PlannerConfig):
    method = method.lower()
    terminal_penalty = env_cfg.cliff_penalty if env_cfg.env_name == "corridor" else env_cfg.break_penalty
    if method == "catso":
        return CATSO(k=4, C=cfg.bonus_C, N=cfg.catso_N, p=cfg.dist_backup_p, tau=cfg.planner_alpha), cfg.dist_backup_p
    if method == "patso":
        return PATSO(k=4, C=cfg.bonus_C, K=cfg.patso_K, p=cfg.dist_backup_p, tau=cfg.planner_alpha), cfg.dist_backup_p
    if method == "scalarts":
        return ScalarTSOptSelector(k=4, C=cfg.scalar_C, p=cfg.dist_backup_p), cfg.dist_backup_p
    if method == "uct":
        uct_c = cfg.uct_c
        if uct_c is None:
            rmin = min(env_cfg.step_cost, env_cfg.step_cost + env_cfg.goal_reward, terminal_penalty)
            rmax = max(env_cfg.step_cost, env_cfg.step_cost + env_cfg.goal_reward, terminal_penalty)
            uct_c = math.sqrt(2.0) * (rmax - rmin)
        return UCTSelector(k=4, ucb_c=uct_c), 1.0
    if method == "poweruct":
        uct_c = cfg.uct_c
        if uct_c is None:
            rmin = min(env_cfg.step_cost, env_cfg.step_cost + env_cfg.goal_reward, terminal_penalty)
            rmax = max(env_cfg.step_cost, env_cfg.step_cost + env_cfg.goal_reward, terminal_penalty)
            uct_c = math.sqrt(2.0) * (rmax - rmin)
        return PowerUCTSelector(k=4, ucb_c=uct_c), cfg.poweruct_p
    raise ValueError(f"Unknown method: {method}")


def evaluate_method(method: str, env_cfg: EnvConfig, cfg: PlannerConfig, oracle) -> Dict[str, np.ndarray]:
    budgets = np.asarray(cfg.budgets, dtype=int)
    seed_records_popt: List[List[float]] = []
    seed_records_sreg: List[List[float]] = []
    seed_records_cvar: List[List[float]] = []

    reward_range = discounted_return_bounds(env_cfg, cfg.gamma)

    for seed in range(cfg.seeds):
        selector, backup_p = make_selector(method, env_cfg, cfg)
        planner = GenericGridMCTS(
            env_factory=lambda sim_seed: make_env(env_cfg, seed=sim_seed),
            root_observation=env_start_observation(env_cfg),
            selector=selector,
            gamma=cfg.gamma,
            reward_range=reward_range,
            backup_p=backup_p,
            recommendation_mode=cfg.recommendation_mode,
            planner_alpha=cfg.planner_alpha,
            seed=seed,
        )

        popt_seed: List[float] = []
        sreg_seed: List[float] = []
        cvar_seed: List[float] = []
        for budget in budgets:
            planner.run_until(int(budget))
            rec_a = planner.recommend_root_action()
            popt_seed.append(1.0 if rec_a in oracle.optimal_actions else 0.0)
            sreg_seed.append(float(oracle.best_q - oracle.Q_root[rec_a]))
            cvar_seed.append(float(oracle.best_root_cvar - oracle.root_cvar[rec_a]))

        seed_records_popt.append(popt_seed)
        seed_records_sreg.append(sreg_seed)
        seed_records_cvar.append(cvar_seed)
        print(f"[{method}] finished seed {seed + 1}/{cfg.seeds}")

    popt = np.asarray(seed_records_popt, dtype=float)
    sreg = np.asarray(seed_records_sreg, dtype=float)
    cvar = np.asarray(seed_records_cvar, dtype=float)
    popt_mean, popt_ci = mean_and_ci(popt)
    sreg_mean, sreg_ci = mean_and_ci(sreg)
    cvar_mean, cvar_ci = mean_and_ci(cvar)

    return {
        "budget": budgets,
        "popt_mean": popt_mean,
        "popt_ci": popt_ci,
        "sreg_mean": sreg_mean,
        "sreg_ci": sreg_ci,
        "cvar_mean": cvar_mean,
        "cvar_ci": cvar_ci,
        "popt_seedwise": popt,
        "sreg_seedwise": sreg,
        "cvar_seedwise": cvar,
    }


def plot_curves(results: Dict[str, Dict[str, np.ndarray]], env_cfg: EnvConfig, cfg: PlannerConfig) -> List[str]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_paths: List[str] = []

    method_order = list(results.keys())
    label = env_label(env_cfg)
    prefix = env_cfg.env_name

    # Figure 1a style: P(optimal action)
    fig = plt.figure(figsize=(7.5, 5.2))
    ax = fig.add_subplot(111)
    for method in method_order:
        r = results[method]
        x = r["budget"]
        y = r["popt_mean"]
        ci = r["popt_ci"]
        ax.plot(x, y, label=method)
        ax.fill_between(x, np.clip(y - ci, 0.0, 1.0), np.clip(y + ci, 0.0, 1.0), alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("# simulations")
    ax.set_ylabel("P(recommend optimal action)")
    ax.set_title(f"{label} | P(optimal start action)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    path1 = os.path.join(cfg.output_dir, f"{prefix}_figure_1a_p_optimal.png")
    fig.tight_layout()
    fig.savefig(path1, dpi=200)
    plt.close(fig)
    out_paths.append(path1)

    # Figure 2a style: simple regret
    fig = plt.figure(figsize=(7.5, 5.2))
    ax = fig.add_subplot(111)
    for method in method_order:
        r = results[method]
        x = r["budget"]
        y = np.maximum(r["sreg_mean"], 1e-12)
        ci = r["sreg_ci"]
        ax.plot(x, y, label=method)
        ax.fill_between(x, np.maximum(y - ci, 1e-12), np.maximum(y + ci, 1e-12), alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("# simulations")
    ax.set_ylabel("simple regret at start state")
    ax.set_title(f"{label} | simple regret at start state")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    path2 = os.path.join(cfg.output_dir, f"{prefix}_figure_2a_simple_regret.png")
    fig.tight_layout()
    fig.savefig(path2, dpi=200)
    plt.close(fig)
    out_paths.append(path2)

    # Extra figure: CVaR regret against the exact root backup law oracle.
    fig = plt.figure(figsize=(7.5, 5.2))
    ax = fig.add_subplot(111)
    for method in method_order:
        r = results[method]
        x = r["budget"]
        y = np.maximum(r["cvar_mean"], 1e-12)
        ci = r["cvar_ci"]
        ax.plot(x, y, label=method)
        ax.fill_between(x, np.maximum(y - ci, 1e-12), np.maximum(y + ci, 1e-12), alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("# simulations")
    ax.set_ylabel(f"CVaR regret (alpha={cfg.eval_cvar_alpha:.3g})")
    ax.set_title(f"{label} | root backup-law CVaR regret")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    path3 = os.path.join(cfg.output_dir, f"{prefix}_figure_3_cvar_regret.png")
    fig.tight_layout()
    fig.savefig(path3, dpi=200)
    plt.close(fig)
    out_paths.append(path3)

    return out_paths


def save_csv(results: Dict[str, Dict[str, np.ndarray]], env_cfg: EnvConfig, cfg: PlannerConfig) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, f"{env_cfg.env_name}_summary_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "budget",
            "p_optimal_mean",
            "p_optimal_ci95",
            "simple_regret_mean",
            "simple_regret_ci95",
            "cvar_regret_mean",
            "cvar_regret_ci95",
        ])
        for method, r in results.items():
            for i, budget in enumerate(r["budget"]):
                writer.writerow([
                    method,
                    int(budget),
                    float(r["popt_mean"][i]),
                    float(r["popt_ci"][i]),
                    float(r["sreg_mean"][i]),
                    float(r["sreg_ci"][i]),
                    float(r["cvar_mean"][i]),
                    float(r["cvar_ci"][i]),
                ])
    return path


def save_config(env_cfg: EnvConfig, cfg: PlannerConfig) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, f"{env_cfg.env_name}_run_config.txt")
    with open(path, "w", encoding="utf-8") as f:
        for k, v in env_cfg.__dict__.items():
            f.write(f"env.{k} = {v}\n")
        for k, v in cfg.__dict__.items():
            f.write(f"planner.{k} = {v}\n")
    return path


def build_oracle(env_cfg: EnvConfig, cfg: PlannerConfig):
    if env_cfg.env_name == "corridor":
        return CorridorOracle(env_cfg, gamma=cfg.gamma, cvar_alpha=cfg.eval_cvar_alpha)
    if env_cfg.env_name == "thinice":
        return ThinIceRolloutOracle(
            env_cfg,
            gamma=cfg.gamma,
            cvar_alpha=cfg.eval_cvar_alpha,
            num_rollouts=cfg.oracle_rollouts,
            seed=0,
        )
    raise ValueError(f"Unknown environment: {env_cfg.env_name}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate CATSO/PATSO on oracle-style stochastic grid environments.")
    # Environment controls
    p.add_argument("--env", choices=["corridor", "thinice"], default="corridor")
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--slip-prob", type=float, default=0.10)
    p.add_argument("--wind-prob", type=float, default=0.20)
    p.add_argument("--thin-frac", type=float, default=0.15)
    p.add_argument("--break-prob", type=float, default=0.10)
    p.add_argument("--step-cost", type=float, default=-1.0)
    p.add_argument("--goal-reward", type=float, default=50.0)
    p.add_argument("--cliff-penalty", type=float, default=-100.0)
    p.add_argument("--break-penalty", type=float, default=-120.0)
    p.add_argument("--max-steps", type=int, default=100)
    # Planner/eval controls
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--planner-alpha", type=float, default=1.0, help="CVaR level used inside CATSO/PATSO. Use 1.0 to match mean-planning figures.")
    p.add_argument("--eval-cvar-alpha", type=float, default=0.10, help="CVaR level used for the extra CVaR-regret figure.")
    p.add_argument("--recommendation-mode", choices=["planner", "mean"], default="planner")
    p.add_argument("--methods", type=str, default="catso,patso,uct,poweruct,scalarts")
    p.add_argument("--budgets", type=str, default="1,2,5,10,20,50,100,200,500,1000,2000,5000,10000")
    p.add_argument("--seeds", type=int, default=100)
    # Hyperparameters from / aligned with the paper appendix grids.
    p.add_argument("--catso-N", type=int, default=100, help="CATSO atoms. Appendix grid includes 10..100; default picks the best-performing endpoint used often in sweeps.")
    p.add_argument("--patso-K", type=int, default=200, help="PATSO particle cap. Appendix sensitivity includes 50,100,200,400.")
    p.add_argument("--bonus-C", type=float, default=8.0, help="Polynomial bonus constant. Appendix grid includes 4,8,16.")
    p.add_argument("--dist-backup-p", type=parse_float_or_inf, default=math.inf, help="Power-mean exponent for CATSO/PATSO V-node backups. Table 9 highlights max backup as best on synthetic trees.")
    p.add_argument("--poweruct-p", type=parse_float_or_inf, default=2.0)
    p.add_argument("--uct-c", type=float, default=None, help="If omitted, uses sqrt(2)*(Rmax-Rmin) from the appendix baseline recipe.")
    p.add_argument("--scalar-C", type=float, default=8.0)
    p.add_argument("--oracle-rollouts", type=int, default=5000, help="Monte Carlo root-action oracle rollouts for environments without the exact corridor DP oracle.")
    p.add_argument("--output-dir", type=str, default="./eval_outputs")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    methods = tuple(m.strip().lower() for m in args.methods.split(",") if m.strip())
    budgets = parse_int_list(args.budgets)
    if not budgets:
        raise ValueError("Provide at least one budget via --budgets.")

    env_cfg = EnvConfig(
        env_name=args.env,
        N=args.N,
        slip_prob=args.slip_prob,
        wind_prob=args.wind_prob,
        thin_frac=args.thin_frac,
        break_prob=args.break_prob,
        step_cost=args.step_cost,
        goal_reward=args.goal_reward,
        cliff_penalty=args.cliff_penalty,
        break_penalty=args.break_penalty,
        max_steps=args.max_steps,
    )
    cfg = PlannerConfig(
        gamma=args.gamma,
        planner_alpha=args.planner_alpha,
        eval_cvar_alpha=args.eval_cvar_alpha,
        recommendation_mode=args.recommendation_mode,
        methods=methods,
        budgets=budgets,
        seeds=args.seeds,
        catso_N=args.catso_N,
        patso_K=args.patso_K,
        bonus_C=args.bonus_C,
        dist_backup_p=args.dist_backup_p,
        poweruct_p=args.poweruct_p,
        uct_c=args.uct_c,
        scalar_C=args.scalar_C,
        oracle_rollouts=args.oracle_rollouts,
        output_dir=args.output_dir,
    )

    oracle = build_oracle(env_cfg, cfg)
    oracle_kind = "exact" if env_cfg.env_name == "corridor" else "Monte Carlo approximation"
    print(f"Environment: {env_label(env_cfg)}")
    print(f"Oracle type: {oracle_kind}")
    print("Oracle root Q*:", oracle.Q_root)
    print("Oracle optimal expected-value actions:", sorted(oracle.optimal_actions))
    print(f"Oracle root backup-law CVaR(alpha={cfg.eval_cvar_alpha}):", oracle.root_cvar)
    print("Oracle optimal CVaR actions:", sorted(oracle.optimal_cvar_actions))

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for method in cfg.methods:
        print(f"\n=== Evaluating {method} ===")
        results[method] = evaluate_method(method, env_cfg, cfg, oracle)

    fig_paths = plot_curves(results, env_cfg, cfg)
    csv_path = save_csv(results, env_cfg, cfg)
    cfg_path = save_config(env_cfg, cfg)

    print("\nSaved files:")
    for path in fig_paths + [csv_path, cfg_path]:
        print(path)


if __name__ == "__main__":
    main()
