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
Run:
  PYTHONPATH=. .venv/bin/python tools/connected_modal_smoke.py --games 1 --rounds 2
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

N_PLAYERS = 5


def _run_agent(name: str, server: str, run_id: str) -> None:
    # cred path must NOT pre-exist (an empty file makes CredentialsStore json.loads("") crash)
    cred = os.path.join(tempfile.mkdtemp(prefix="arena-cred-"), "cred.json")
    try:
        agent = ArenaAgent(name=name, server=server, credentials=CredentialsStore(cred))
        agent.act(random_agent.act)
        signup = agent.signup(run_id=run_id)
        print(f"  {name}: {signup.status} seat={signup.seat}", flush=True)
        agent.run_forever([signup])
        print(f"  {name}: done", flush=True)
    except Exception as e:
        print(f"  {name}: ERROR {type(e).__name__}: {e}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None)
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--seed", type=int, default=31337)
    p.add_argument("--url-timeout", type=float, default=180.0)
    args = p.parse_args()

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
