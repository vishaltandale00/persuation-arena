"""Cross-run rating engine — objective-handicap Elo on the per-seat win condition.

The unit is `game_players.won` (the per-seat objective outcome), NOT a binary team result:
`compute_winners` resolves three independent objective groups (village / werewolf / tanner), so a
Tanner can co-win with the village and both factions can lose. Each seat *i* dealt role *r* updates
its competitor's skill by

    pᵢ = σ(sᵢ − d_r − ρᵢ)              (expected: own skill, role difficulty, resistance)
    sᵢ ← sᵢ + k·dampᵢ·(wonᵢ − pᵢ)

where d_r is the population-shared role-difficulty handicap (−logit of the role's base win-rate) and
ρᵢ is the mean current skill of the seats opposing *i*'s objective.

Ratings are DERIVED: `recompute()` replays every game in canonical order
`(run.created_utc, run_id, gid)`, all pᵢ in a game computed from pre-game skills and applied after,
so a rebuild from scratch is a pure function of the stored games — two recomputes are byte-identical.
Run with `python -m arena.rating recompute`.
"""
from __future__ import annotations

import argparse
import math

from . import store
from .games.base import is_no_contest

# --- tunables (mirror GOAL-rating-leaderboard.md §Tunables) -----------------------------------
K_BASE = 0.08            # base skill step, in logits
RD0 = 0.60               # initial rating deviation (logits); wide → fast early convergence
RD_MIN = 0.10            # rating-deviation floor for established competitors
K_RD_CAP = 4.0           # cap on the provisional step multiplier (rd/RD_MIN)
PROVISIONAL_GAMES = 30   # below this many games a competitor is flagged provisional
BETA_ALPHA = 2.0         # Beta prior for role-difficulty base rates (keeps rare roles finite)
MIN_ROLE_TRIALS = 30     # a (bucket, role) d_r cell needs this many trials before it's trusted
ELO_BASE = 1500.0
ELO_SCALE = 173.0        # ≈ 400 / ln 10; converts logit skill to a familiar Elo number
_EPS = 1e-9


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(1.0 - _EPS, max(_EPS, p))
    return math.log(p / (1.0 - p))


def objective_group(dealt_role: str | None, team: str | None) -> str:
    """The seat's objective group. Tanner is its own group; otherwise it's the seat's faction.
    Mirrors `player_won`: Tanner wins by dying, the werewolf team shares one objective, the rest
    share the village objective."""
    if dealt_role == "Tanner":
        return "tanner"
    return "werewolf" if team == "evil" else "village"


def _resists(my_group: str, other_group: str) -> bool:
    """Does a seat in `other_group` oppose my objective?"""
    if my_group == "village":
        return other_group == "werewolf"
    if my_group == "werewolf":
        return other_group == "village"
    return True  # Tanner is resisted by the whole table (it needs to be voted out)


def _bucket(game: str | None, players, deck) -> str:
    return f"{game}|{players}|{deck if deck is not None else '-'}"


# --- role difficulty (d_r), Level 0: pooled base rate + Beta prior, bucketed with fallback -----

def compute_difficulty(rows: list[dict], run_meta: dict[str, dict], stamp: str):
    """Return (storage_rows, lookup) where lookup(game, players, deck, role) -> d_r.

    d_r = −logit((w+α)/(n+2α)). Cells are computed at three granularities — full
    (game|players|deck), game-level, and global ("*") — and the finest cell with ≥ MIN_ROLE_TRIALS
    trials wins at lookup, falling back to coarser buckets (and finally 0)."""
    agg: dict[str, dict[str, list[int]]] = {}

    def bump(bucket: str, role: str, won: int):
        cell = agg.setdefault(bucket, {}).setdefault(role, [0, 0])
        cell[0] += won
        cell[1] += 1

    # group into games so no-contest games (no evil seat, no winner) drop out of the base rates too.
    by_game: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        by_game.setdefault((r["run_id"], r["gid"]), []).append(r)
    for seats in by_game.values():
        if is_no_contest(seats):
            continue
        for r in seats:
            rm = run_meta.get(r["run_id"], {})
            role = r.get("dealt_role") or "?"
            won = int(r.get("won") or 0)
            bump(_bucket(rm.get("game"), rm.get("players"), rm.get("deck_preset")), role, won)
            bump(f"{rm.get('game')}", role, won)
            bump("*", role, won)

    diff: dict[str, dict[str, dict]] = {}
    storage: list[dict] = []
    for bucket, roles in agg.items():
        for role, (w, n) in roles.items():
            base = (w + BETA_ALPHA) / (n + 2 * BETA_ALPHA)
            d_r = -_logit(base)
            diff.setdefault(bucket, {})[role] = {"n": n, "d_r": d_r}
            storage.append({"bucket": bucket, "role": role, "w": w, "n": n,
                            "base_rate": base, "d_r": d_r, "updated_utc": stamp})
    storage.sort(key=lambda d: (d["bucket"], d["role"]))

    def lookup(game, players, deck, role: str) -> float:
        for b in (_bucket(game, players, deck), f"{game}", "*"):
            cell = diff.get(b, {}).get(role)
            if cell and cell["n"] >= MIN_ROLE_TRIALS:
                return cell["d_r"]
        glob = diff.get("*", {}).get(role)
        return glob["d_r"] if glob else 0.0

    return storage, lookup


def _rd_for(games: int) -> float:
    return max(RD_MIN, RD0 / math.sqrt(1 + games))


# --- the replay -------------------------------------------------------------------------------

def _identity(row: dict, roster_harness: dict[tuple[str, str], str]) -> tuple[str, str | None, str, str | None, str | None]:
    """-> (identity_key, agent_id, display_name, declared_model, declared_harness).
    Connected seats key on agent_id (the bearer-token identity); static roster seats key on
    model:harness so a model is rated even without a token."""
    agent_id = row.get("agent_id")
    if agent_id:
        return agent_id, agent_id, row.get("agent") or agent_id, None, None
    model = row.get("model") or "?"
    harness = roster_harness.get((row["run_id"], row.get("agent")), "base")
    return f"static:{model}:{harness}", None, model, model, harness


def recompute() -> dict:
    """Replay all games, write role_difficulty + rating_events + ratings, return a summary."""
    rows = store.all_game_player_rows()
    run_meta = store.run_meta_map()
    agents_by_id = {a["id"]: a for a in store.list_agents()}

    # static roster harness lookup: (run_id, agent_name) -> harness
    roster_harness: dict[tuple[str, str], str] = {}
    for rid, rm in run_meta.items():
        for a in rm.get("agents", []):
            if a.get("name"):
                roster_harness[(rid, a["name"])] = a.get("harness") or "base"

    # data-derived stamp so the snapshot is a pure function of the games (deterministic).
    stamp = max((rm.get("created_utc") or "") for rm in run_meta.values()) if run_meta else ""

    difficulty_rows, dr_lookup = compute_difficulty(rows, run_meta, stamp)

    # group rows into games, ordered canonically
    games: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        games.setdefault((r["run_id"], r["gid"]), []).append(r)

    def order_key(k):
        rid, gid = k
        return ((run_meta.get(rid, {}).get("created_utc") or ""), rid, gid)

    skills: dict[str, float] = {}
    gcount: dict[str, int] = {}
    wins: dict[str, int] = {}
    calls_sum: dict[str, int] = {}
    forfeits_sum: dict[str, int] = {}
    meta: dict[str, dict] = {}   # identity_key -> {agent_id, display_name, declared_model, declared_harness}
    events: list[dict] = []
    rated_games = 0

    for key in sorted(games.keys(), key=order_key):
        seats = sorted(games[key], key=lambda r: r["seat"])
        if is_no_contest(seats):
            continue  # no opposing faction in play — not a rateable contest
        rated_games += 1
        rid, gid = key
        rm = run_meta.get(rid, {})
        # resolve identities + groups once (pre-game snapshot)
        info = []
        for r in seats:
            ident, agent_id, disp, dmodel, dharness = _identity(r, roster_harness)
            group = objective_group(r.get("dealt_role"), r.get("team"))
            info.append({"row": r, "ident": ident, "group": group})
            m = meta.setdefault(ident, {"agent_id": agent_id, "display_name": disp,
                                        "declared_model": dmodel, "declared_harness": dharness})
            if agent_id and agent_id in agents_by_id:  # connected: agents table is authoritative
                a = agents_by_id[agent_id]
                m["display_name"] = a.get("display_name") or disp
                m["declared_model"] = a.get("declared_model")
                m["declared_harness"] = a.get("declared_harness")

        deltas: list[tuple[str, float]] = []
        for idx, this in enumerate(info):
            r = this["row"]
            ident, group = this["ident"], this["group"]
            s_i = skills.get(ident, 0.0)
            rd = _rd_for(gcount.get(ident, 0))
            resisting = [o for j, o in enumerate(info) if j != idx and _resists(group, o["group"])]
            rho = (sum(skills.get(o["ident"], 0.0) for o in resisting) / len(resisting)
                   if resisting else 0.0)
            d_r = dr_lookup(rm.get("game"), rm.get("players"), rm.get("deck_preset"),
                            r.get("dealt_role") or "?")
            p = _sigmoid(s_i - d_r - rho)
            calls, forf = int(r.get("calls") or 0), int(r.get("forfeits") or 0)
            damp = max(0.0, 1.0 - (forf / calls if calls else 0.0))
            k_eff = K_BASE * min(K_RD_CAP, rd / RD_MIN) * damp
            won = int(r.get("won") or 0)
            delta = k_eff * (won - p)
            deltas.append((ident, delta))
            events.append({
                "id": f"re_{rid}_{gid}_{r['seat']}", "identity_key": ident, "run_id": rid,
                "gid": gid, "seat": r["seat"], "dealt_role": r.get("dealt_role"),
                "objective_group": group, "won": won, "pre_skill": s_i, "post_skill": s_i + delta,
                "delta": delta, "expected": p, "d_r": d_r, "resistance": rho, "k": k_eff,
                "damped": damp, "created_utc": stamp,
            })

        for idx, this in enumerate(info):
            r, ident = this["row"], this["ident"]
            skills[ident] = skills.get(ident, 0.0) + deltas[idx][1]
            gcount[ident] = gcount.get(ident, 0) + 1
            wins[ident] = wins.get(ident, 0) + int(r.get("won") or 0)
            calls_sum[ident] = calls_sum.get(ident, 0) + int(r.get("calls") or 0)
            forfeits_sum[ident] = forfeits_sum.get(ident, 0) + int(r.get("forfeits") or 0)

    ratings: list[dict] = []
    for ident in sorted(skills.keys()):
        g = gcount.get(ident, 0)
        rd = _rd_for(g)
        s = skills[ident]
        c, f = calls_sum.get(ident, 0), forfeits_sum.get(ident, 0)
        m = meta.get(ident, {})
        ratings.append({
            "identity_key": ident, "agent_id": m.get("agent_id"),
            "display_name": m.get("display_name"), "declared_model": m.get("declared_model"),
            "declared_harness": m.get("declared_harness"),
            "skill": s, "rd": rd, "elo": ELO_BASE + ELO_SCALE * s,
            "games": g, "wins": wins.get(ident, 0),
            "forfeit_rate": round(f / c, 4) if c else 0.0,
            "provisional": 1 if g < PROVISIONAL_GAMES else 0, "updated_utc": stamp,
        })

    events.sort(key=lambda e: (e["created_utc"], e["run_id"], e["gid"], e["seat"]))
    store.replace_ratings(difficulty_rows, events, ratings)
    return {"games": rated_games, "competitors": len(ratings), "events": len(events),
            "difficulty_cells": len(difficulty_rows)}


# --- API assembly: descriptive per-role/objective breakdown on top of the snapshot -----------

def _conservative(row: dict) -> float:
    return row["elo"] - 2 * ELO_SCALE * row["rd"]


def _cell(w: int, n: int) -> dict:
    from .score import wilson
    lo, hi, p = wilson(w, n)
    return {"w": w, "n": n, "rate": round(p, 3), "lo": round(lo, 3), "hi": round(hi, 3)}


def _aggregate(events: list[dict], diff_global: dict[str, dict]):
    """Roll a competitor's rating_events into overall / by_objective / by_role cells. by_role cells
    carry the role's population base rate, d_r, and the competitor's vs-spread (rate − base)."""
    overall = [0, 0]
    by_obj: dict[str, list[int]] = {}
    by_role: dict[str, list[int]] = {}
    for e in events:
        won = int(e.get("won") or 0)
        overall[0] += won
        overall[1] += 1
        og = e.get("objective_group") or "village"
        c = by_obj.setdefault(og, [0, 0]); c[0] += won; c[1] += 1
        role = e.get("dealt_role") or "?"
        c = by_role.setdefault(role, [0, 0]); c[0] += won; c[1] += 1

    by_role_out = {}
    for role, (w, n) in sorted(by_role.items()):
        cell = _cell(w, n)
        d = diff_global.get(role) or {}
        base, dr = d.get("base_rate"), d.get("d_r")
        cell["base"] = round(base, 3) if base is not None else None
        cell["d_r"] = round(dr, 3) if dr is not None else None
        cell["vs_spread"] = round(cell["rate"] - base, 3) if base is not None else None
        cell["hard"] = bool(dr is not None and dr > 0)
        by_role_out[role] = cell

    by_obj_out = {k: _cell(*v) for k, v in by_obj.items()}
    for k in ("village", "werewolf", "tanner"):
        by_obj_out.setdefault(k, _cell(0, 0))
    return _cell(*overall), by_obj_out, by_role_out


def _public(row: dict) -> dict:
    """Snapshot scalar fields safe to serialize (no token ever lives here anyway)."""
    return {
        "identity_key": row["identity_key"], "agent_id": row["agent_id"],
        "display_name": row["display_name"], "declared_model": row["declared_model"],
        "declared_harness": row["declared_harness"],
        "elo": round(row["elo"], 1), "rd": round(173.0 * row["rd"], 1),
        "conservative": round(_conservative(row), 1),
        "games": row["games"], "wins": row["wins"],
        "provisional": bool(row["provisional"]), "forfeit_rate": row["forfeit_rate"],
    }


def _diff_global() -> dict[str, dict]:
    return {role: cell for (bucket, role), cell in store.role_difficulty_map().items() if bucket == "*"}


def leaderboard() -> list[dict]:
    """Ranked competitors with the full per-role breakdown for the UI (already conservatively sorted)."""
    diff_global = _diff_global()
    events_by: dict[str, list[dict]] = {}
    for e in store.all_rating_events():
        events_by.setdefault(e["identity_key"], []).append(e)
    out = []
    for row in store.leaderboard_rows():
        overall, by_obj, by_role = _aggregate(events_by.get(row["identity_key"], []), diff_global)
        out.append({**_public(row), "overall": overall, "by_objective": by_obj, "by_role": by_role})
    return out


def agent_detail(identity_key: str) -> dict | None:
    """One competitor's scorecard + rating history + the runs it played. Accepts an agent_id
    (connected identity) or any identity_key."""
    row = store.get_rating(identity_key)
    if not row:
        return None
    events = store.rating_events_for(identity_key)
    overall, by_obj, by_role = _aggregate(events, _diff_global())
    history = [{
        "run_id": e["run_id"], "gid": e["gid"], "objective_group": e["objective_group"],
        "dealt_role": e["dealt_role"], "won": e["won"],
        "pre_elo": round(ELO_BASE + ELO_SCALE * e["pre_skill"], 1),
        "post_elo": round(ELO_BASE + ELO_SCALE * e["post_skill"], 1),
        "delta_elo": round(ELO_SCALE * e["delta"], 1), "expected": round(e["expected"], 3),
    } for e in events]
    runs = sorted({e["run_id"] for e in events})
    return {**_public(row), "overall": overall, "by_objective": by_obj, "by_role": by_role,
            "history": history, "runs": runs}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="arena.rating")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("recompute", help="rebuild ratings from all stored games")
    top = sub.add_parser("top", help="recompute then print the leaderboard")
    top.add_argument("-n", type=int, default=10)
    args = parser.parse_args(argv)

    if args.cmd in (None, "recompute", "top"):
        summary = recompute()
        print(f"recompute: {summary}")
        if args.cmd == "top":
            rows = store.leaderboard_rows()[: args.n]
            for i, r in enumerate(rows, 1):
                flag = " (prov)" if r["provisional"] else ""
                print(f"{i:>2}. {r['elo']:>7.0f} ±{173*r['rd']:>3.0f}  n={r['games']:>3}  "
                      f"{r['identity_key']}{flag}")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
