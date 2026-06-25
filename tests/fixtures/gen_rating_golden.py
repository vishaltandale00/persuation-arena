"""Generate a DETERMINISTIC golden fixture for parity-testing a JS port of arena/rating.py.

Builds a fresh temp SQLite store, saves hand-specified games across 3 runs, runs
arena.rating.recompute(), then dumps BOTH the recompute INPUTS and OUTPUTS to a JSON
fixture, with everything sorted for stable diffs.

Run with:  ARENA_STORE_DIR=<tmp> .venv/bin/python gen_rating_golden.py <fixture_path>
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Make sure we use SQLite, not Postgres.
os.environ.pop("DATABASE_URL", None)

# Point the SQLite store at a fresh temp file BEFORE importing store (config reads
# ARENA_STORE_DIR at import). We also re-pin store.DB_PATH after import to be safe.
_tmpdir = tempfile.mkdtemp(prefix="rating_golden_")
os.environ["ARENA_STORE_DIR"] = _tmpdir

from arena import rating, store  # noqa: E402

# config.py runs load_dotenv(.env) at import, which can re-inject DATABASE_URL (routing to
# Postgres). Force SQLite: drop it AFTER import so _is_pg() is False for every conn() call.
os.environ.pop("DATABASE_URL", None)
assert not store._is_pg(), "expected SQLite backend; DATABASE_URL still set"

# Fresh DB file in the temp dir.
store.DB_PATH = Path(_tmpdir) / "arena.db"
if store.DB_PATH.exists():
    store.DB_PATH.unlink()
store.init_schema()


def save_run(run_id, players, agents, deck="classic", created="2026-01-01T00:00:00Z", game="onuw"):
    store.save_run({
        "id": run_id, "game": game, "label": run_id.upper(), "status": "done",
        "n_games": 99, "players": players, "seed_base": 1, "created": created[:16],
        "agents": agents, "created_utc": created, "deck_preset": deck,
    })


def save_game(run_id, gid, seats, agents, winner_team="good"):
    """seats: list of {seat, role, team, won, [calls], [forfeits]} (contiguous).
    agents: list aligned to seat index -> {name, model, [harness], [agent_id], [signup_id]}."""
    seats = sorted(seats, key=lambda s: s["seat"])
    transcript = {
        "seed": gid, "winner_team": winner_team, "outcome": {"text": f"{run_id}#{gid}"},
        "players": [{"seat": s["seat"], "dealt": s["role"], "end": s["role"],
                     "team": s["team"], "won": bool(s["won"]),
                     "calls": s.get("calls", 4), "forfeits": s.get("forfeits", 0)}
                    for s in seats],
    }
    store.save_game(run_id, gid, transcript, agents)


# --------------------------------------------------------------------------------------------
# Run A — STATIC roster, two models on the 'base' harness. ONUW 5-seat, deck 'classic'.
# Covers static:{model}:{harness} identities, common roles + the rare Tanner role.
# agents are positional by seat; harness comes from the run roster (run_meta agents).
# --------------------------------------------------------------------------------------------
A_AGENTS = [
    {"name": "n0", "model": "gpt-x", "harness": "base"},
    {"name": "n1", "model": "claude-y", "harness": "base"},
    {"name": "n2", "model": "gpt-x", "harness": "base"},
    {"name": "n3", "model": "claude-y", "harness": "base"},
    {"name": "n4", "model": "gpt-x", "harness": "base"},
]
save_run("runA", 5, A_AGENTS, deck="classic", created="2026-01-01T00:00:00Z")

# Six hand-specified games. Roles rotate; outcomes alternate so both factions win sometimes.
# Seat->role fixed; the seat at index 4 is the rare Tanner in a couple of games.
A_GAMES = [
    # gid 0: village wins, wolf loses, tanner loses
    [("Werewolf", "evil", 0), ("Villager", "good", 1), ("Seer", "good", 1), ("Robber", "good", 1), ("Tanner", "good", 0)],
    # gid 1: wolf wins, village loses, tanner loses
    [("Werewolf", "evil", 1), ("Villager", "good", 0), ("Seer", "good", 0), ("Robber", "good", 0), ("Tanner", "good", 0)],
    # gid 2: tanner co-wins WITH village; wolf loses
    [("Werewolf", "evil", 0), ("Villager", "good", 1), ("Seer", "good", 1), ("Robber", "good", 1), ("Tanner", "good", 1)],
    # gid 3: village wins; a forfeiting seer gains less
    [("Werewolf", "evil", 0), ("Villager", "good", 1), ("Seer", "good", 1, 4, 3), ("Robber", "good", 1), ("Troublemaker", "good", 1)],
    # gid 4: wolf wins
    [("Werewolf", "evil", 1), ("Villager", "good", 0), ("Seer", "good", 0), ("Robber", "good", 0), ("Troublemaker", "good", 0)],
    # gid 5: village wins
    [("Werewolf", "evil", 0), ("Villager", "good", 1), ("Seer", "good", 1), ("Robber", "good", 1), ("Troublemaker", "good", 1)],
]
for gid, rows in enumerate(A_GAMES):
    seats = []
    for seat, spec in enumerate(rows):
        role, team, won = spec[0], spec[1], spec[2]
        calls = spec[3] if len(spec) > 3 else 4
        forf = spec[4] if len(spec) > 4 else 0
        seats.append({"seat": seat, "role": role, "team": team, "won": won,
                      "calls": calls, "forfeits": forf})
    save_game("runA", gid, seats, A_AGENTS)

# --------------------------------------------------------------------------------------------
# Run B — STATIC roster, introduces gpt-x on a DIFFERENT harness ('turbo') so it is a distinct
# identity static:gpt-x:turbo, alongside the existing claude-y:base. ONUW 4-seat, deck 'classic'
# (so the (game|players|deck) bucket differs from Run A's 5-seat bucket).
# --------------------------------------------------------------------------------------------
B_AGENTS = [
    {"name": "b0", "model": "gpt-x", "harness": "turbo"},
    {"name": "b1", "model": "claude-y", "harness": "base"},
    {"name": "b2", "model": "gpt-x", "harness": "turbo"},
    {"name": "b3", "model": "claude-y", "harness": "base"},
]
save_run("runB", 4, B_AGENTS, deck="classic", created="2026-01-02T00:00:00Z")
B_GAMES = [
    # gid 0: village wins
    [("Werewolf", "evil", 0), ("Villager", "good", 1), ("Seer", "good", 1), ("Mason", "good", 1)],
    # gid 1: wolf wins
    [("Werewolf", "evil", 1), ("Villager", "good", 0), ("Seer", "good", 0), ("Mason", "good", 0)],
    # gid 2: tanner co-wins with village
    [("Werewolf", "evil", 0), ("Villager", "good", 1), ("Tanner", "good", 1), ("Mason", "good", 1)],
]
for gid, rows in enumerate(B_GAMES):
    seats = [{"seat": seat, "role": r, "team": t, "won": w, "calls": 4, "forfeits": 0}
             for seat, (r, t, w) in enumerate(rows)]
    save_game("runB", gid, seats, B_AGENTS)

# --------------------------------------------------------------------------------------------
# Run C — CONNECTED run. One registered agent (agent_id -> agents table is authoritative for its
# display_name/declared_model/declared_harness) plays a couple games -> a PROVISIONAL competitor
# keyed on agent_id, NOT on model:harness. The other seats are static gpt-x:base (reused identity).
# --------------------------------------------------------------------------------------------
agent = store.register_agent(
    display_name="Connected One",
    token_hash="hash_connected_one",
    protocol_version="1.0",
    sdk_version="0.1.0",
    agent_id="agent_connected_one",
    declared_model="claude-y",
    declared_harness="connected-sdk",
)
CONN_ID = agent["id"]

# register_agent stamps wall-clock created/last_seen times. recompute() never reads them (it only
# uses id/display_name/declared_model/declared_harness), but they live in the input list_agents, so
# pin them to a fixed value to keep the fixture byte-reproducible.
with store.conn() as _c:
    _c.execute(
        "UPDATE agents SET created_utc=?, last_seen_utc=? WHERE id=?",
        ("2026-01-03T00:00:00Z", "2026-01-03T00:00:00Z", CONN_ID),
    )

C_AGENTS = [
    {"name": "Connected One", "model": "connected-agent", "harness": "connected",
     "agent_id": CONN_ID, "signup_id": "su_conn_1"},
    {"name": "c1", "model": "gpt-x", "harness": "base"},
    {"name": "c2", "model": "gpt-x", "harness": "base"},
]
save_run("runC", 3, C_AGENTS, deck="classic", created="2026-01-03T00:00:00Z")
C_GAMES = [
    # gid 0: connected villager wins, wolf loses
    [("Villager", "good", 1), ("Werewolf", "evil", 0), ("Seer", "good", 1)],
    # gid 1: connected villager loses, wolf wins
    [("Villager", "good", 0), ("Werewolf", "evil", 1), ("Seer", "good", 0)],
]
for gid, rows in enumerate(C_GAMES):
    seats = [{"seat": seat, "role": r, "team": t, "won": w, "calls": 4, "forfeits": 0}
             for seat, (r, t, w) in enumerate(rows)]
    save_game("runC", gid, seats, C_AGENTS)

# --------------------------------------------------------------------------------------------
# Recompute the ratings (the function under parity test).
# --------------------------------------------------------------------------------------------
summary = rating.recompute()

# --------------------------------------------------------------------------------------------
# Collect INPUTS (a) and OUTPUTS (b), sorted for stable diffs.
# --------------------------------------------------------------------------------------------
def sort_rows(rows, keyfn):
    return sorted(rows, key=keyfn)

inputs = {
    "all_game_player_rows": sort_rows(
        store.all_game_player_rows(),
        lambda r: (r["run_id"], r["gid"], r["seat"]),
    ),
    # run_meta_map is a dict keyed by run_id; JSON object keys preserve insertion -> sort keys.
    "run_meta_map": dict(sorted(store.run_meta_map().items())),
    "list_agents": sort_rows(store.list_agents(), lambda a: a["id"]),
}

# role_difficulty_map() returns {(bucket, role): cell}; tuple keys aren't JSON-serializable, so
# emit a sorted list of cells (each carries bucket+role).
role_diff = [
    {"bucket": bucket, "role": role, **{k: v for k, v in cell.items()}}
    for (bucket, role), cell in store.role_difficulty_map().items()
]
role_diff = sort_rows(role_diff, lambda d: (d["bucket"], d["role"]))

outputs = {
    "leaderboard_rows": sort_rows(
        store.leaderboard_rows(),
        # leaderboard_rows is already conservatively ordered; re-sort by identity_key so the
        # fixture is order-stable independent of float tie-break. JS test re-sorts the same way.
        lambda r: r["identity_key"],
    ),
    "all_rating_events": sort_rows(
        store.all_rating_events(),
        lambda e: (e["created_utc"], e["run_id"], e["gid"], e["seat"]),
    ),
    "role_difficulty_map": role_diff,
}

fixture = {
    "_meta": {
        "description": "Golden parity fixture for arena/rating.py recompute(). "
                       "Feed `inputs` to the JS compute port and assert it reproduces `outputs`.",
        "source": "arena.rating.recompute()",
        "recompute_summary": summary,
        "tunables": {
            "K_BASE": rating.K_BASE, "RD0": rating.RD0, "RD_MIN": rating.RD_MIN,
            "K_RD_CAP": rating.K_RD_CAP, "PROVISIONAL_GAMES": rating.PROVISIONAL_GAMES,
            "BETA_ALPHA": rating.BETA_ALPHA, "MIN_ROLE_TRIALS": rating.MIN_ROLE_TRIALS,
            "ELO_BASE": rating.ELO_BASE, "ELO_SCALE": rating.ELO_SCALE,
        },
    },
    "inputs": inputs,
    "outputs": outputs,
}

out_path = Path(sys.argv[1])
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    json.dump(fixture, f, indent=2, sort_keys=True, ensure_ascii=False)
    f.write("\n")

print(f"wrote {out_path}")
print(f"summary: {summary}")
print(f"competitors: {[r['identity_key'] for r in outputs['leaderboard_rows']]}")
print(f"events: {len(outputs['all_rating_events'])}  diff_cells: {len(outputs['role_difficulty_map'])}")
