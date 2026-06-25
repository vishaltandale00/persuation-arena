"""End-to-end smoke test for the Modal per-run coordinator (arena/modal_app.py).

Spawns a per-run Modal container, waits for its Tunnel URL, connects N scripted agents directly to it,
lets the game(s) play, and reports the container's result. Uses the FREE scripted random_agent, so it
proves the TOPOLOGY — per-run server + tunnel + agent connectivity + coordination + Neon persistence —
with no LLM cost.

The Modal container does all Neon writes (creates the run, serves, coordinates, persists); this process
only spawns + drives agents, so it never queries prod Neon directly.

Prereqs:
  .venv/bin/modal deploy arena/modal_app.py
  modal secret `neon-database-url` holding DATABASE_URL
Run (single coordinator):
  PYTHONPATH=. .venv/bin/python tools/connected_modal_smoke.py --games 1 --rounds 2

V-10 — manual parallel-shards gate (SPEC-parallel-shards.md REQ-10). Spawns K real coordinator
containers and K shard host bundles (process per shard, D9), each agent identity signed into ALL K
shards via a SHARED cred file (one rating competitor across shards, D9). Requires a deployed Modal
app + DATABASE_URL on the Modal side; NOT part of CI.
  # deployed Modal + DATABASE_URL set
  PYTHONPATH=. .venv/bin/python tools/connected_modal_smoke.py --shards 2 --games 4 --rounds 2
Assertions to eyeball (REQ-10): both children reach `running` before either reaches `done`
(structural concurrency); final rollup_parent_status == 'done'; the union of games == N; and the
observer renders the parent as one run (its overview aggregates the children).
"""
from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time

import modal

from persuasion_arena_agent.agent import ArenaAgent
from persuasion_arena_agent.credentials import CredentialsStore
from examples import random_agent
from arena.modal_app import APP_NAME, coordinator_url
from arena.sharded import create_sharded_run, rollup_parent_status

N_PLAYERS = 5


def _run_agent(name: str, server: str, run_id: str, cred: str | None = None) -> None:
    # cred path must NOT pre-exist (an empty file makes CredentialsStore json.loads("") crash). A
    # SHARED cred path (reused across shards) makes one identity a single competitor across shards.
    cred = cred or os.path.join(tempfile.mkdtemp(prefix="arena-cred-"), "cred.json")
    try:
        agent = ArenaAgent(name=name, server=server, credentials=CredentialsStore(cred))
        agent.act(random_agent.act)
        signup = agent.signup(run_id=run_id)
        print(f"  {name}: {signup.status} seat={signup.seat}", flush=True)
        agent.run_forever([signup])
        print(f"  {name}: done", flush=True)
    except Exception as e:
        print(f"  {name}: ERROR {type(e).__name__}: {e}", flush=True)


def _shared_creds(n: int) -> list[str]:
    """One cred path per identity, reused across ALL shards (shared creds => one competitor)."""
    return [os.path.join(tempfile.mkdtemp(prefix="arena-shardcred-"), f"id_{i}.json")
            for i in range(n)]


def _sharded_smoke(args) -> int:
    """V-10 (manual): K real coordinator containers + K shard host bundles, parent rolled up on read.

    Each of the N identities is signed into EVERY shard via a SHARED cred file, so it is one rating
    competitor across shards (D9). The parent is presentational; rollup_parent_status computes its
    state on read (D1). Requires DATABASE_URL set on the Modal side so the containers persist to Neon.
    """
    parent_id = args.run_id or f"modal_smoke_shards_{args.seed}"
    parent_cfg = {
        "id": parent_id, "game": "onuw", "label": "One Night Ultimate Werewolf",
        "status": "open", "n_games": args.games, "players": N_PLAYERS,
        "seed_base": args.seed, "submitter": "modal-smoke", "deck_preset": "arena",
    }
    child_ids = create_sharded_run(parent_cfg, args.shards)
    print(f"parent {parent_id} -> {len(child_ids)} shard(s): {', '.join(child_ids)}", flush=True)

    cred_paths = _shared_creds(N_PLAYERS)  # shared across shards: one competitor per identity

    fn = modal.Function.from_name(APP_NAME, "run_server")
    calls = {}
    for cid in child_ids:
        child_cfg = {**parent_cfg, "id": cid}
        calls[cid] = fn.spawn(child_cfg, args.rounds, args.url_timeout)
        print(f"  spawned coordinator for {cid}, call_id={calls[cid].object_id}", flush=True)

    threads = []
    for cid in child_ids:
        deadline = time.time() + args.url_timeout
        url = None
        while time.time() < deadline:
            url = coordinator_url(cid)
            if url:
                break
            time.sleep(1.0)
        if not url:
            raise SystemExit(f"no coordinator URL for {cid} — check modal app logs")
        print(f"  {cid} coordinator URL: {url}", flush=True)
        for i in range(N_PLAYERS):
            # shared cred file per identity across shards (one rating competitor, D9).
            t = threading.Thread(target=_run_agent,
                                 args=(f"rand-{i}", url, cid, cred_paths[i]), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(0.3)

    for t in threads:
        t.join(timeout=600)

    status = rollup_parent_status(parent_id)
    print(f"parent {parent_id} rollup status = {status}", flush=True)
    for cid in child_ids:
        try:
            print(f"  {cid} result: {calls[cid].get(timeout=120)}", flush=True)
        except Exception as e:
            print(f"  {cid} could not get result: {type(e).__name__}: {e}", flush=True)
    return 0 if status == "done" else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None)
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--seed", type=int, default=31337)
    p.add_argument("--url-timeout", type=float, default=180.0)
    p.add_argument("--shards", type=int, default=1,
                   help="V-10 parallel-shards gate (SPEC-parallel-shards.md REQ-10): spawn K real "
                        "coordinator containers + K shard host bundles, each identity signed into ALL "
                        "K shards via a shared cred file. Default 1 = today's single-coordinator smoke.")
    args = p.parse_args()

    if args.shards and args.shards > 1:
        return _sharded_smoke(args)

    run_id = args.run_id or f"modal_smoke_{args.seed}"
    run_config = {
        "id": run_id, "game": "onuw", "label": "One Night Ultimate Werewolf",
        "status": "open", "n_games": args.games, "players": N_PLAYERS,
        "seed_base": args.seed, "submitter": "modal-smoke", "deck_preset": "arena",
    }

    print(f"spawning Modal coordinator for {run_id} ...", flush=True)
    fn = modal.Function.from_name(APP_NAME, "run_server")
    call = fn.spawn(run_config, args.rounds, args.url_timeout)
    print(f"  spawned, call_id={call.object_id}", flush=True)

    print("waiting for tunnel URL ...", flush=True)
    url = None
    deadline = time.time() + args.url_timeout
    while time.time() < deadline:
        url = coordinator_url(run_id)
        if url:
            break
        time.sleep(1.0)
    if not url:
        raise SystemExit("no coordinator URL — check: .venv/bin/modal app logs persuasion-arena-coordinator")
    print(f"  coordinator URL: {url}", flush=True)

    print(f"connecting {N_PLAYERS} scripted agents ...", flush=True)
    threads = []
    for i in range(N_PLAYERS):
        t = threading.Thread(target=_run_agent, args=(f"rand-{i}", url, run_id), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.3)  # stagger so seat order is stable
    for t in threads:
        t.join(timeout=600)

    print("waiting for coordinator result ...", flush=True)
    try:
        result = call.get(timeout=120)
        print(f"RESULT: {result}", flush=True)
    except Exception as e:
        print(f"could not get result: {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
