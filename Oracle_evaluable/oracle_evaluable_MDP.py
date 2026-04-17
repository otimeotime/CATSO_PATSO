from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random
import sys

# ---------------------------------------------------------------------
# 1) Risky-Shortcut Gridworld (Windy Cliff)
# ---------------------------------------------------------------------

Coord = Tuple[int, int]

@dataclass
class WindSpec:
    wind_prob: float = 0.20
    windy_cols: Optional[List[int]] = None

class RiskyShortcutGridworld:
    """
    A cliff-walking grid with a short, risky corridor (wind pushes toward cliff).
    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    Observation: (row, col) or single index if observation_mode="index"
    """

    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    ACTIONS = (UP, RIGHT, DOWN, LEFT)
    DELTAS = {UP:(-1,0), RIGHT:(0,1), DOWN:(1,0), LEFT:(0,-1)}

    def __init__(self,
                 N: int = 10,
                 slip_prob: float = 0.10,
                 wind: Optional[WindSpec] = None,
                 step_cost: float = -1.0,
                 goal_reward: float = +50.0,
                 cliff_penalty: float = -100.0,
                 observation_mode: str = "tuple",
                 max_steps: Optional[int] = None,
                 seed: Optional[int] = None,
                 color_render: bool = False):
        assert N >= 4
        assert observation_mode in ("tuple","index")
        self.N = N
        self.slip_prob = float(slip_prob)
        self.step_cost = float(step_cost)
        self.goal_reward = float(goal_reward)
        self.cliff_penalty = float(cliff_penalty)
        self.observation_mode = observation_mode
        self.max_steps = max_steps if max_steps is not None else 10*N*N
        self.color_render = bool(color_render)

        self.start: Coord = (N-1, 0)
        self.goal:  Coord = (N-1, N-1)
        self.cliff_cells = {(N-1, c) for c in range(1, N-1)}

        w = wind or WindSpec()
        self.wind_prob = float(w.wind_prob)
        self.windy_cols = list(range(1, N-1)) if w.windy_cols is None else list(w.windy_cols)

        self._rng = random.Random(seed)
        self._pos: Coord = self.start
        self._steps = 0
        self._terminated = False
        self._truncated = False

    def reset(self, seed: Optional[int]=None, options: Optional[Dict[str,Any]]=None):
        if seed is not None:
            self._rng.seed(seed)
        self._pos = self.start
        self._steps = 0
        self._terminated = False
        self._truncated = False
        return self._obs_from_pos(self._pos), {"catastrophic": False, "slipped": False, "wind": False}

    def step(self, action: int):
        assert action in self.ACTIONS
        if self._terminated or self._truncated:
            raise RuntimeError("Episode is done. Call reset().")

        self._steps += 1
        slipped = False
        wind_applied = False
        catastrophic = False

        if self._rng.random() < self.slip_prob:
            action = self._rng.choice(self.ACTIONS)
            slipped = True

        r, c = self._pos
        dr, dc = self.DELTAS[action]
        nr, nc = self._clip(r+dr, c+dc)

        if nc in self.windy_cols and self._rng.random() < self.wind_prob:
            nr, nc = self._clip(nr+1, nc)
            wind_applied = True

        terminated = False
        reward = self.step_cost
        if (nr, nc) in self.cliff_cells:
            reward = self.cliff_penalty
            terminated = True
            catastrophic = True
        elif (nr, nc) == self.goal:
            reward = self.step_cost + self.goal_reward
            terminated = True

        self._pos = (nr, nc)

        truncated = False
        if not terminated and self._steps >= self.max_steps:
            truncated = True

        self._terminated = terminated
        self._truncated = truncated
        obs = self._obs_from_pos(self._pos)
        info = {"catastrophic": catastrophic, "slipped": slipped, "wind": wind_applied, "pos": self._pos, "steps": self._steps}
        return obs, reward, terminated, truncated, info

    def legal_actions(self, state: Optional[Coord]=None) -> List[int]:
        return list(self.ACTIONS)

    def render(self, file=sys.stdout):
        grid = [["." for _ in range(self.N)] for _ in range(self.N)]
        for (rr,cc) in self.cliff_cells: grid[rr][cc]="X"
        sr,sc = self.start; gr,gc = self.goal
        grid[sr][sc] = "S"; grid[gr][gc]="G"
        ar,ac = self._pos; grid[ar][ac] = "A"
        print("\n".join(" ".join(row) for row in grid), file=file); print(file=file)

    def _clip(self, r:int, c:int) -> Coord:
        return (max(0,min(self.N-1,r)), max(0,min(self.N-1,c)))

    def _obs_from_pos(self, pos: Coord):
        return pos if self.observation_mode=="tuple" else (pos[0]*self.N + pos[1])

    # Snapshot helpers
    def get_state(self):
        return (self._pos, self._steps, self._terminated, self._truncated, self._rng.getstate())
    def set_state(self, state):
        self._pos, self._steps, self._terminated, self._truncated, rng_state = state
        self._rng.setstate(rng_state)



#####################################################################################################
class ThinIceFrozenLakePlus:
    """
    Grid with hidden 'thin-ice' tiles: stepping on them may catastrophically break.
    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    """
    UP, RIGHT, DOWN, LEFT = 0,1,2,3
    ACTIONS = (UP, RIGHT, DOWN, LEFT)
    DELTAS = {UP:(-1,0), RIGHT:(0,1), DOWN:(1,0), LEFT:(0,-1)}

    def __init__(self,
                 N:int=8,
                 slip_prob: float=0.15,
                 thin_frac: float=0.15,      # fraction of non-start/goal tiles that are thin
                 break_prob: float=0.10,     # failure prob when stepping on thin ice
                 step_cost: float=-1.0,
                 goal_reward: float=+40.0,
                 break_penalty: float=-120.0,
                 observation_mode: str="tuple",
                 max_steps: Optional[int]=None,
                 seed: Optional[int]=None):
        assert N>=4
        assert observation_mode in ("tuple","index")
        self.N=N
        self.slip_prob=float(slip_prob)
        self.thin_frac=thin_frac
        self.break_prob=float(break_prob)
        self.step_cost=float(step_cost)
        self.goal_reward=float(goal_reward)
        self.break_penalty=float(break_penalty)
        self.observation_mode=observation_mode
        self.max_steps = max_steps if max_steps is not None else 10*N*N

        self.start=(0,0); self.goal=(N-1,N-1)
        self._rng = random.Random(seed)

        # Sample thin-ice tiles (hidden) excluding start/goal
        all_cells=[(r,c) for r in range(N) for c in range(N) if (r,c) not in (self.start,self.goal)]
        k = max(1, int(round(self.thin_frac*len(all_cells))))
        self.thin=set(self._rng.sample(all_cells, k))

        self._pos=self.start; self._steps=0
        self._terminated=False; self._truncated=False

    def reset(self, seed: Optional[int]=None, options: Optional[Dict[str,Any]]=None):
        if seed is not None: self._rng.seed(seed)
        # re-sample thin map to keep episodes independent if desired
        N=self.N
        all_cells=[(r,c) for r in range(N) for c in range(N) if (r,c) not in (self.start,self.goal)]
        k = max(1, int(round(self.thin_frac*len(all_cells))))
        self.thin=set(self._rng.sample(all_cells, k))

        self._pos=self.start; self._steps=0
        self._terminated=False; self._truncated=False
        return self._obs(), {"thin_known": False}

    def step(self, action:int):
        assert action in self.ACTIONS
        if self._terminated or self._truncated:
            raise RuntimeError("Episode done. Call reset().")
        self._steps += 1

        if self._rng.random() < self.slip_prob:
            action = self._rng.choice(self.ACTIONS)

        r,c = self._pos
        dr,dc = self.DELTAS[action]
        nr,nc = self._clip(r+dr,c+dc)

        terminated=False; truncated=False; catastrophic=False
        reward=self.step_cost

        if (nr,nc)==self.goal:
            reward = self.step_cost + self.goal_reward
            terminated=True
        elif (nr,nc) in self.thin:
            if self._rng.random() < self.break_prob:
                reward = self.break_penalty
                terminated=True
                catastrophic=True

        self._pos=(nr,nc)

        if not terminated and self._steps>=self.max_steps:
            truncated=True

        self._terminated=terminated; self._truncated=truncated
        return self._obs(), reward, terminated, truncated, {
            "catastrophic": catastrophic,
            "pos": self._pos,
            "steps": self._steps
        }

    def legal_actions(self, state: Optional[Coord]=None)->List[int]:
        return list(self.ACTIONS)

    def render(self, file=sys.stdout):
        grid=[["." for _ in range(self.N)] for _ in range(self.N)]
        sr,sc=self.start; gr,gc=self.goal
        grid[sr][sc]="S"; grid[gr][gc]="G"
        ar,ac=self._pos; grid[ar][ac]="A"
        # thin tiles hidden; uncomment to visualize:
        # for (r,c) in self.thin: grid[r][c] = "t" if grid[r][c]=="." else grid[r][c]
        print("\n".join(" ".join(row) for row in grid), file=file); print(file=file)

    def _clip(self,r:int,c:int)->Coord:
        return (max(0,min(self.N-1,r)), max(0,min(self.N-1,c)))

    def _obs(self):
        return self._pos if self.observation_mode=="tuple" else (self._pos[0]*self.N + self._pos[1])

    # Snapshot
    def get_state(self):
        return (self._pos, self._steps, self._terminated, self._truncated, tuple(sorted(self.thin)), self._rng.getstate())
    def set_state(self, state):
        self._pos, self._steps, self._terminated, self._truncated, thin_tuple, rng_state = state
        self.thin=set(thin_tuple); self._rng.setstate(rng_state)