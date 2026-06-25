"""Parallel sharding orchestration (SPEC-parallel-shards.md §3, §6.4-6.5).

A logical run of N games can execute as K concurrent child runs ("shards"), each a normal
connected run playing a disjoint stride slice (`g % K == k`) of one global schedule, presented to
the world as a single parent run.

Topology (D1): decentralized. `create_sharded_run` writes the parent (`run_kind='parent'`) and K
children (`run_kind='child'`), each carrying its shard coordinates and the GLOBAL N in `n_games`
(D8). No process owns the parent; its status is computed-on-read by `rollup_parent_status`.

`rating.py` is untouched (INV-3): children are normal runs to the rating engine, keyed on
`agent_id`; shared creds across shards make one competitor (REQ-5).
"""
from __future__ import annotations

import os

from . import store

# K is explicit (default 1) with a hard cap (D3). The cap is tunable via env; default 4.
DEFAULT_MAX_SHARDS = 4


def max_shards() -> int:
    """Hard cap on K (SPEC D3). Tunable via ARENA_MAX_SHARDS; default 4."""
    try:
        cap = int(os.environ.get("ARENA_MAX_SHARDS", DEFAULT_MAX_SHARDS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_SHARDS
    return cap if cap >= 1 else DEFAULT_MAX_SHARDS


def rollup_parent_status(parent_id: str) -> str:
    """Compute the parent's status from its children (SPEC §6.4 / REQ-2 / V-2).

    - no children started yet -> 'open'
    - all children 'done' -> 'done'
    - any child 'partial' or otherwise failed -> 'partial'
    - any child not yet done (e.g. 'running'/'open'/'waiting') -> 'running'
    """
    child_ids = store.child_run_ids(parent_id)
    if not child_ids:
        return "open"
    statuses = []
    for cid in child_ids:
        run = store.get_run(cid)
        statuses.append((run or {}).get("status") or "open")
    if any(s == "partial" for s in statuses):
        return "partial"
    if all(s == "done" for s in statuses):
        return "done"
    return "running"


def create_sharded_run(parent_config: dict, num_shards: int) -> list[str]:
    """Write the parent + K child runs for a sharded run; return the child run ids in shard order.

    `parent_config` carries the GLOBAL run fields: `id`, `game`, `label`, `n_games` (global N),
    `players`, `seed_base`, and optionally `deck_preset`, `agents`, `metadata`, `created`,
    `created_utc`. K is validated to `1 <= num_shards <= max_shards()` (D3); above the cap is
    rejected.

    The parent row is `run_kind='parent'` (sole writer is this orchestrator, D2) and stores the
    global N in `n_games` and `num_shards`. Each child is created via `create_connected_run` with
    `run_kind='child'`, `parent_run_id`, `shard_index`, `num_shards`, and the GLOBAL `n_games`
    (the runner slices to its shard via `fresh_deal_schedule`, D8). All start `status='open'`.
    """
    if not isinstance(num_shards, int) or isinstance(num_shards, bool):
        raise ValueError(f"num_shards must be an int, got {num_shards!r}")
    cap = max_shards()
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if num_shards > cap:
        raise ValueError(f"num_shards {num_shards} exceeds cap {cap}")

    parent_id = parent_config["id"]
    n_games = int(parent_config["n_games"])  # GLOBAL N (D8)
    players = int(parent_config["players"])
    seed_base = int(parent_config["seed_base"])
    game = parent_config["game"]
    label = parent_config.get("label", game)
    deck_preset = parent_config.get("deck_preset")
    agents = parent_config.get("agents", [])
    metadata = parent_config.get("metadata") or {}
    created = parent_config.get("created")
    created_utc = parent_config.get("created_utc")

    # Parent: a presentational umbrella; never discoverable/joinable (INV-4 via list_open_runs).
    store.save_run({
        "id": parent_id,
        "game": game,
        "label": label,
        "status": "open",
        "n_games": n_games,
        "players": players,
        "seed_base": seed_base,
        "created": created or (created_utc or store._utcnow())[:16].replace("T", " "),
        "created_utc": created_utc or store._utcnow(),
        "submitter": parent_config.get("submitter", "sharded"),
        "agents": agents,
        "deck_preset": deck_preset,
        "metadata": metadata,
        "run_kind": "parent",
        "parent_run_id": None,
        "shard_index": None,
        "num_shards": num_shards,
    })

    child_ids: list[str] = []
    for k in range(num_shards):
        child_id = f"{parent_id}_shard_{k}"
        store.create_connected_run({
            "id": child_id,
            "game": game,
            "label": label,
            "status": "open",
            "n_games": n_games,  # GLOBAL N; the runner slices to this shard (D8)
            "players": players,
            "seed_base": seed_base,
            "created": created,
            "created_utc": created_utc,
            "submitter": parent_config.get("submitter", "sharded"),
            "agents": agents,
            "deck_preset": deck_preset,
            "metadata": metadata,
            "run_kind": "child",
            "parent_run_id": parent_id,
            "shard_index": k,
            "num_shards": num_shards,
        })
        child_ids.append(child_id)
    return child_ids
