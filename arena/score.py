"""Role-balanced win-rate with 95% confidence intervals.

For ONUW the cleanest faction split is village (good) vs werewolf-team (evil), since dealt roles
mutate during the night. We report overall plus per-faction win-rate, each with a Wilson 95% CI
(robust at small/zero counts, unlike the normal approximation).

We also report a per-role breakdown keyed on the **dealt** role (the role the agent was handed and
had to navigate — the unit the role-balancing schedule equalizes), and a per-agent **forfeit rate**
(turns where the model failed and the engine substituted a default action). Forfeits are non-skill
noise; a high rate means that agent's win-rate is partly driven by defaulted actions, not play.
"""
from __future__ import annotations

import math

from . import store


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (lo, hi, point) for k successes in n trials. point = k/n."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half), p)


def _cell(w: int, n: int) -> dict:
    lo, hi, p = wilson(w, n)
    return {"w": w, "n": n, "rate": round(p, 3), "lo": round(lo, 3), "hi": round(hi, 3)}


def score_run(run_id: str) -> dict:
    """agent -> scorecard.

    Each scorecard has {w,n,rate,lo,hi} cells for `overall`, `good`, `evil`, plus a `by_role`
    map of {dealt_role: cell} (the realized per-role histogram is visible as each cell's n), and
    reliability fields {calls, forfeits, forfeit_rate}.
    """
    rows = store.player_rows(run_id)
    agg: dict[str, dict] = {}
    for r in rows:
        a = agg.setdefault(r["agent"], {
            "overall": [0, 0], "good": [0, 0], "evil": [0, 0],
            "by_role": {}, "calls": 0, "forfeits": 0,
        })
        won = r["won"]
        a["overall"][1] += 1
        a["overall"][0] += won
        t = r["team"] if r["team"] in ("good", "evil") else "good"
        a[t][1] += 1
        a[t][0] += won
        role = r.get("dealt_role") or "?"
        cell = a["by_role"].setdefault(role, [0, 0])
        cell[1] += 1
        cell[0] += won
        a["calls"] += r.get("calls") or 0
        a["forfeits"] += r.get("forfeits") or 0

    out: dict[str, dict] = {}
    for agent, d in agg.items():
        calls, forfeits = d["calls"], d["forfeits"]
        out[agent] = {
            "overall": _cell(*d["overall"]),
            "good": _cell(*d["good"]),
            "evil": _cell(*d["evil"]),
            "by_role": {role: _cell(w, n) for role, (w, n) in sorted(d["by_role"].items())},
            "calls": calls,
            "forfeits": forfeits,
            "forfeit_rate": round(forfeits / calls, 3) if calls else 0.0,
        }
    return out
