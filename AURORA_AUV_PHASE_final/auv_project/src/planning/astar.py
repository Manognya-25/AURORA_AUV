"""Grid A* over a PlanningGrid (pure NumPy/stdlib; no PyBullet, no truth).

Cost model: the cost of a move between adjacent cells a -> b is  step_length_m * 0.5 * (cost[a] + cost[b]),
so the optimum is the cost-weighted shortest path through the Phase-4 cost map (clearance and unknown-space
penalties included). Occupied / inflated / (policy-blocked unknown) / out-of-bounds cells are never expanded.

Admissibility: every step costs at least step_length * c_min, c_min = the cheapest traversable cell cost, so
h = c_min * (geometric lower bound) never overestimates (octile/euclidean for 8-connectivity; manhattan only
for 4-connectivity, otherwise it is silently replaced by octile and reported). 8-connected moves forbid
corner cutting: a diagonal step needs BOTH adjacent orthogonal cells traversable.
Deterministic: ties are broken by insertion order, neighbours are visited in a fixed order.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from src.planning.grid_utils import PlanningGrid

_SQRT2 = math.sqrt(2.0)
# (drow, dcol, length_in_cells); fixed order => deterministic
_N4 = ((0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0))
_N8 = _N4 + ((1, 1, _SQRT2), (1, -1, _SQRT2), (-1, -1, _SQRT2), (-1, 1, _SQRT2))


@dataclass
class AStarResult:
    found: bool
    cells: list = field(default_factory=list)      # [(row, col), ...] start..goal inclusive
    cost: float = float("inf")
    expansions: int = 0
    reason: str = ""                               # "" | no_path | start_blocked | goal_blocked | expansion_limit | out_of_bounds
    heuristic_used: str = "octile"


def _heuristic_fn(name: str, connectivity: int, res: float, h_scale: float, gr: int, gc: int):
    if connectivity == 8 and name == "manhattan":
        name = "octile"
    if name == "manhattan":
        return (lambda r, c: h_scale * res * (abs(r - gr) + abs(c - gc))), name
    if name == "euclidean":
        return (lambda r, c: h_scale * res * math.hypot(r - gr, c - gc)), name

    def octile(r, c):
        dr, dc = abs(r - gr), abs(c - gc)
        return h_scale * res * ((dr + dc) + (_SQRT2 - 2.0) * min(dr, dc))
    if connectivity == 4:
        # octile is a valid (weaker) lower bound for 4-connectivity too
        pass
    return octile, "octile"


def astar(grid: PlanningGrid, start: tuple, goal: tuple, connectivity: int = 8, heuristic: str = "octile",
          heuristic_weight: float = 1.0, max_expansions: int = 600000) -> AStarResult:
    rows, cols = grid.shape
    sr, sc = start
    gr, gc = goal
    for (r, c, why) in ((sr, sc, "start"), (gr, gc, "goal")):
        if not (0 <= r < rows and 0 <= c < cols):
            return AStarResult(False, reason="out_of_bounds")
    if not grid.traversable[sr, sc]:
        return AStarResult(False, reason="start_blocked")
    if not grid.traversable[gr, gc]:
        return AStarResult(False, reason="goal_blocked")

    res = grid.resolution
    trav = grid.traversable.reshape(-1).tolist()
    cost = grid.cost.reshape(-1).tolist()
    n = rows * cols
    W = cols
    s_idx, g_idx = sr * W + sc, gr * W + gc
    h_scale = grid.min_traversable_cost() * float(heuristic_weight)
    hfn, hname = _heuristic_fn(heuristic, connectivity, res, h_scale, gr, gc)
    nbrs = _N8 if connectivity == 8 else _N4

    INF = float("inf")
    g = [INF] * n
    parent = [-1] * n
    closed = bytearray(n)
    g[s_idx] = 0.0
    heap = [(hfn(sr, sc), 0, s_idx)]
    counter = 1
    expansions = 0
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        f, _, cur = pop(heap)
        if closed[cur]:
            continue
        closed[cur] = 1
        if cur == g_idx:
            break
        expansions += 1
        if expansions > max_expansions:
            return AStarResult(False, expansions=expansions, reason="expansion_limit", heuristic_used=hname)
        r, c = divmod(cur, W)
        gcur, ccur = g[cur], cost[cur]
        for dr, dc, ln in nbrs:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            nidx = nr * W + nc
            if not trav[nidx] or closed[nidx]:
                continue
            if dr != 0 and dc != 0 and not (trav[r * W + nc] and trav[nr * W + c]):
                continue                                              # no corner cutting
            ng = gcur + ln * res * 0.5 * (ccur + cost[nidx])
            if ng < g[nidx]:
                g[nidx] = ng
                parent[nidx] = cur
                push(heap, (ng + hfn(nr, nc), counter, nidx))
                counter += 1
    if g[g_idx] == INF:
        return AStarResult(False, expansions=expansions, reason="no_path", heuristic_used=hname)
    cells = []
    k = g_idx
    while k != -1:
        cells.append(divmod(k, W))
        k = parent[k]
    cells.reverse()
    return AStarResult(True, cells, g[g_idx], expansions, "", hname)
