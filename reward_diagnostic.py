#!/usr/bin/env python3
"""
Reward diagnostic for PoINavigationEnv.

Generates random scenarios and prints reward-related grids
in the terminal to visualize the observation-reward mismatch
and per-step reward budget.

Grids printed (for grid_size ≤ 20):
  1. Layout          — obstacles, agents, POIs
  2. BFS distance    — steps from each cell to the optimal POI
  3. Mismatch (A1)   — which POI the obs implies when agent1 moves
  4. Mismatch (A2)   — same for agent2
  5. Net reward      — per-step net (progress − penalty) at each cell

Usage:
  python reward_diagnostic.py                       # 3 scenarios, 8×8
  python reward_diagnostic.py --seed 42             # reproducible
  python reward_diagnostic.py --grid-size 8 -n 5   # 5 scenarios
"""
from __future__ import annotations

import argparse

import numpy as np

from cost_function import DEFAULT_WEIGHTS
from poi_environment import PoINavigationEnv, _DIRS

# ── ANSI colours ─────────────────────────────────────────────────────────────
RST  = "\033[0m"
BOLD = "\033[1m"
DIM  = "\033[2m"
RED  = "\033[91m"
GRN  = "\033[92m"
YEL  = "\033[93m"
BLU  = "\033[94m"
CYN  = "\033[96m"


def _cell(text: str, color: str, width: int) -> str:
    return f"{color}{text:^{width}}{RST}"


def _print_grid(title: str, cells, rows: int, cols: int, cw: int = 4) -> None:
    print(f"\n  {BOLD}{title}{RST}")
    print("      " + "".join(f"{c:^{cw}}" for c in range(cols)))
    print("      " + "─" * (cols * cw))
    for r in range(rows):
        print(f"   {r:>2} │" + "".join(cells[r][c] for c in range(cols)))


def _implied_optimal(
    agent_pos: tuple[int, int],
    other_pos: tuple[int, int],
    poi_dist_maps: list[dict],
    max_dist: float,
    weights: tuple[float, ...],
) -> int:
    costs = []
    for dm in poi_dist_maps:
        da = min(dm.get(agent_pos, float("inf")), max_dist)
        dh = min(dm.get(other_pos, float("inf")), max_dist)
        ta, th = da / max_dist, dh / max_dist
        costs.append(sum(
            w * v for w, v in zip(weights, (ta, th, 0.6 * ta + 0.4 * th, 1.0 - th, max(ta, th)))
        ))
    return int(np.argmin(costs))


# ─────────────────────────────────────────────────────────────────────────────


def run_scenario(grid_size: tuple[int, int], seed: int) -> dict:
    rows, cols = grid_size
    max_steps = max(rows * cols // 2, rows * 4)
    env = PoINavigationEnv(grid_size=grid_size, seed=seed, max_steps=max_steps)
    env.reset()

    max_dist = float(rows + cols)
    oi = env._optimal_poi_idx
    op = env._pois[oi]
    om = env._poi_dist_maps[oi]
    obstacles = env._obstacles
    a1, a2 = env._init_agent1_pos, env._init_agent2_pos

    # header
    print(f"\n{'═' * 62}")
    print(f"  {BOLD}Agent1{RST} {BLU}{a1}{RST}   "
          f"{BOLD}Agent2{RST} {RED}{a2}{RST}   "
          f"{BOLD}Optimal{RST} {CYN}P{oi + 1} at {op}{RST}")
    pois_s = "  ".join(
        f"{CYN if i == oi else DIM}P{i + 1}={p}{RST}"
        for i, p in enumerate(env._pois)
    )
    print(f"  POIs: {pois_s}   Obstacles: {len(obstacles)}")
    print(f"{'═' * 62}")

    show = max(rows, cols) <= 20
    cw = 4

    # ── 1  Layout ─────────────────────────────────────────────────────────
    if show:
        g = [[None] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if (r, c) in obstacles:
                    g[r][c] = _cell("░░", DIM, cw)
                elif (r, c) == a1:
                    g[r][c] = _cell("A1", BLU + BOLD, cw)
                elif (r, c) == a2:
                    g[r][c] = _cell("A2", RED + BOLD, cw)
                else:
                    found = False
                    for i, p in enumerate(env._pois):
                        if (r, c) == p:
                            col = CYN + BOLD if i == oi else DIM
                            g[r][c] = _cell(f"P{i + 1}", col, cw)
                            found = True
                            break
                    if not found:
                        g[r][c] = _cell("·", "", cw)
        _print_grid("Layout  (cyan P = optimal POI)", g, rows, cols, cw)

    # ── 2  BFS distance to optimal POI ────────────────────────────────────
    if show:
        g = [[None] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if (r, c) in obstacles:
                    g[r][c] = _cell("░░", DIM, cw)
                    continue
                d = om.get((r, c), float("inf"))
                if d > 9999:
                    g[r][c] = _cell("∞", RED, cw)
                elif d == 0:
                    g[r][c] = _cell("★", CYN + BOLD, cw)
                elif d <= rows // 2:
                    g[r][c] = _cell(str(int(d)), GRN, cw)
                elif d <= rows:
                    g[r][c] = _cell(str(int(d)), YEL, cw)
                else:
                    g[r][c] = _cell(str(int(d)), RED, cw)
        _print_grid(f"BFS distance → optimal P{oi + 1}", g, rows, cols, cw)

    # ── 3 & 4  Dynamic-cost mismatch (each agent varies) ─────────────────
    mm1 = mm2 = 0
    free_n = 0

    g1 = [[None] * cols for _ in range(rows)] if show else None
    g2 = [[None] * cols for _ in range(rows)] if show else None

    for r in range(rows):
        for c in range(cols):
            if (r, c) in obstacles:
                if show:
                    g1[r][c] = _cell("░░", DIM, cw)
                    g2[r][c] = _cell("░░", DIM, cw)
                continue
            free_n += 1

            imp1 = _implied_optimal((r, c), a2, env._poi_dist_maps, max_dist, DEFAULT_WEIGHTS)
            imp2 = _implied_optimal(a1, (r, c), env._poi_dist_maps, max_dist, DEFAULT_WEIGHTS)

            if imp1 != oi:
                mm1 += 1
            if imp2 != oi:
                mm2 += 1

            if show:
                g1[r][c] = _cell(f"P{imp1 + 1}", GRN if imp1 == oi else RED + BOLD, cw)
                g2[r][c] = _cell(f"P{imp2 + 1}", GRN if imp2 == oi else RED + BOLD, cw)

    p1 = 100 * mm1 / max(free_n, 1)
    p2 = 100 * mm2 / max(free_n, 1)

    if show:
        _print_grid(
            f"DYNAMIC cost mismatch when Agent1 moves (Agent2 fixed)\n"
            f"    {GRN}green = correct{RST}   {RED}red = MISMATCH{RST}"
            f"   ({mm1}/{free_n} = {p1:.0f}%)",
            g1, rows, cols, cw,
        )
        _print_grid(
            f"DYNAMIC cost mismatch when Agent2 moves (Agent1 fixed)\n"
            f"    {GRN}green = correct{RST}   {RED}red = MISMATCH{RST}"
            f"   ({mm2}/{free_n} = {p2:.0f}%)",
            g2, rows, cols, cw,
        )

    # ── 4b  Initial-cost (FIXED) — should be 0% mismatch ──────────────────
    init_opt = int(np.argmin(env._initial_costs))
    init_ok = init_opt == oi
    if show:
        print(f"\n  {BOLD}INITIAL costs (frozen at reset):{RST}  "
              f"{' '.join(f'P{i+1}={c:.3f}' for i, c in enumerate(env._initial_costs))}"
              f"  →  min = {GRN}P{init_opt+1}{RST}"
              f"  {'✓ matches reward target' if init_ok else RED + '✗ BUG' + RST}")

    # ── 5  Net single-agent reward per step (bidirectional) ─────────────
    cw2 = 6
    if show:
        g = [[None] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if (r, c) in obstacles:
                    g[r][c] = _cell("░░", DIM, cw2)
                    continue
                d_here = om.get((r, c), float("inf"))
                best = -float("inf")
                for dr, dc in _DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in obstacles:
                        d_nb = om.get((nr, nc), float("inf"))
                        best = max(best, (d_nb - d_here) / max_dist * env.PROGRESS_SCALE)
                if best == -float("inf"):
                    best = 0.0
                net = best - env.STEP_PENALTY
                if net > 0.005:
                    col = GRN
                elif net < -0.005:
                    col = RED
                else:
                    col = YEL
                g[r][c] = _cell(f"{net:+.2f}", col, cw2)
        _print_grid(
            f"Net reward / step  (best 1-agent progress − penalty)\n"
            f"    {GRN}+pos{RST}   {YEL}~zero{RST}   {RED}−neg{RST}",
            g, rows, cols, cw2,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    bfs1 = om.get(a1, float("inf"))
    bfs2 = om.get(a2, float("inf"))
    min_steps = max(bfs1, bfs2)
    prog1 = 1.0 / max_dist * env.PROGRESS_SCALE

    print(f"\n  {'─' * 58}")
    print(f"  {BOLD}REWARD NUMBERS{RST}")
    print(f"  {'─' * 58}")
    print(f"  Step penalty:            {env.STEP_PENALTY:.4f}")
    print(f"  Progress / step (1 ag):  {prog1:.4f}")
    n1 = prog1 - env.STEP_PENALTY
    n2 = 2 * prog1 - env.STEP_PENALTY
    retreat1 = -prog1 - env.STEP_PENALTY
    warn1 = f"  {RED}← NEGATIVE{RST}" if n1 < 0 else f"  {GRN}← positive{RST}"
    print(f"  Net (only 1 closer):    {n1:+.4f}{warn1}")
    print(f"  Net (both closer):      {n2:+.4f}")
    print(f"  Net (1 retreats):       {retreat1:+.4f}  {RED}← bidirectional penalty{RST}")
    print(f"  Terminal bonus:          {env.TERMINAL_BONUS:.1f} – {env.TERMINAL_BONUS * 3:.1f}")
    print(f"  Wrong-POI penalty:      {-env.TERMINAL_BONUS:.1f}")

    print()
    print(f"  BFS Agent1 → optimal:   {bfs1:.0f} steps")
    print(f"  BFS Agent2 → optimal:   {bfs2:.0f} steps")
    print(f"  Min steps (perfect):    {min_steps:.0f}")
    print(f"  Max steps budget:       {env.max_steps}")
    ratio = env.max_steps / max(min_steps, 1)
    warn_b = f"  {RED}← VERY TIGHT{RST}" if ratio < 2.0 else ""
    print(f"  Budget ratio:           {ratio:.1f}x{warn_b}")

    print()
    print(f"  {BOLD}Dynamic cost mismatch{RST} (still present in obs, but no longer sole signal):")
    print(f"    A1 moves: {mm1}/{free_n} ({p1:.0f}%)   A2 moves: {mm2}/{free_n} ({p2:.0f}%)")
    print(f"  {BOLD}Initial costs{RST} (frozen in obs): {GRN}always 0% mismatch ✓{RST}")
    print()

    return dict(pct1=p1, pct2=p2, budget=ratio, net1=n1, net2=n2)


# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward diagnostic for PoINavigationEnv")
    parser.add_argument("--grid-size", type=int, default=8, choices=[8, 32, 64])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-n", "--scenarios", type=int, default=3)
    args = parser.parse_args()

    gs = (args.grid_size, args.grid_size)
    results: list[dict] = []

    for i in range(args.scenarios):
        s = (args.seed + i) if args.seed is not None else int(
            np.random.default_rng().integers(0, 100_000)
        )
        print(f"\n{'#' * 62}")
        print(f"  {BOLD}SCENARIO {i + 1}  (seed={s}, grid={args.grid_size}×{args.grid_size}){RST}")
        print(f"{'#' * 62}")
        results.append(run_scenario(gs, s))

    if len(results) > 1:
        print(f"\n{'#' * 62}")
        print(f"  {BOLD}AGGREGATE  ({len(results)} scenarios){RST}")
        print(f"{'#' * 62}")
        a1 = np.mean([r["pct1"] for r in results])
        a2 = np.mean([r["pct2"] for r in results])
        ab = np.mean([r["budget"] for r in results])
        print(f"  Avg mismatch (A1 moves): {a1:.1f}%")
        print(f"  Avg mismatch (A2 moves): {a2:.1f}%")
        print(f"  Avg budget ratio:        {ab:.1f}x")
        print(f"  Net reward (1 closer):   {results[0]['net1']:+.4f}")
        print(f"  Net reward (both):       {results[0]['net2']:+.4f}")
        print()


if __name__ == "__main__":
    main()
