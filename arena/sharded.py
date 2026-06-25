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
import secrets

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

    # One per-parent secret, shared by the parent and every child (INV-4 / SPEC D7). A child is
    # joinable ONLY by an agent that presents this token; a stray agent posting the predictable
    # parent id or `{parent}_shard_0` child id never has it. The parent itself is never joinable.
    # On a retry/re-create for an existing parent, REUSE the persisted token rather than minting a
    # fresh one — save_run upserts join_token=COALESCE(excluded, existing), so a new token would
    # rotate the secret out from under any token an orchestrator already handed to agents.
    existing = store.get_run(parent_id)
    # A parent_id that already belongs to a NORMAL or CHILD run must NOT be rewritten as a sharded
    # parent (FINDING #1, DATA-INTEGRITY): num_shards is None for a normal run, so the shard-count
    # guard below would be skipped and save_run would clobber that run's row into run_kind='parent',
    # hiding its games/signups (parent read paths source only child runs). A reused --run-id or
    # payload run_id triggers this; reject it loudly. Only an existing run_kind=='parent' is a legal
    # re-create target (the same-K idempotent retry path; a different K is rejected just below).
    if existing is not None and ((existing.get("run_kind") or "normal") != "parent"):
        raise ValueError(
            f"run id {parent_id!r} is already in use by a non-parent run "
            f"(run_kind={existing.get('run_kind') or 'normal'!r}); refusing to overwrite it"
        )
    # Re-creating an existing parent with a DIFFERENT num_shards is not supported: shrinking K would
    # only upsert the new range and leave stale orphan children (e.g. {parent}_shard_2) still
    # pointing at parent_run_id, so child_run_ids() and every parent rollup/score keep counting them.
    # We do NOT silently delete rows; reject the shard-count change. A re-create with the SAME K
    # stays idempotent (the round-4 retry path reuses the token and re-upserts the same children).
    existing_k = (existing or {}).get("num_shards")
    if existing_k is not None and int(existing_k) != num_shards:
        raise ValueError(
            f"shard-count change not supported for parent {parent_id!r}: "
            f"existing num_shards={existing_k}, requested {num_shards}"
        )
    join_token = (existing or {}).get("join_token") or secrets.token_urlsafe(24)

    # FINDING #1 (round-7, DATA-INTEGRITY): the parent-id guard above protects {parent}, but the
    # DERIVED child ids ({parent}_shard_k) are upserted blindly by create_connected_run below. If a
    # NORMAL run already exists named like "{parent}_shard_0" (a reused id, or a parent whose name
    # collides with another run's id), that create rewrites the existing row into a child of THIS new
    # parent — hiding/corrupting the old run and folding its games into the new parent's rollup.
    # Validate FIRST (before any write): each derived child id must be either absent OR already a
    # child of THIS parent (run_kind=='child' and parent_run_id==parent_id). A collision with any
    # pre-existing non-child run, or a child of a DIFFERENT parent, raises and writes nothing new.
    # The same-K idempotent re-create (children already belong to this parent) passes unchanged.
    for k in range(num_shards):
        child_id = f"{parent_id}_shard_{k}"
        prior = store.get_run(child_id)
        if prior is None:
            continue
        prior_kind = prior.get("run_kind") or "normal"
        prior_parent = prior.get("parent_run_id")
        if not (prior_kind == "child" and prior_parent == parent_id):
            raise ValueError(
                f"derived shard id {child_id!r} is already in use by a "
                f"{prior_kind!r} run"
                + (f" (parent_run_id={prior_parent!r})" if prior_parent is not None else "")
                + f"; refusing to overwrite it as a shard of {parent_id!r}"
            )

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
        "join_token": join_token,
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
            "join_token": join_token,
        })
        child_ids.append(child_id)
    return child_ids
