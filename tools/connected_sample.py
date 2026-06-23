"""Orchestrate a connected-agent sample run end to end.

Creates an OPEN connected run, connects N real model-backed agents over the SDK/HTTP using the
reference stateful harnesses, waits for them to seat + ready, coordinates the games to completion,
and prints the role-balanced scorecard.

This is the "connected" path (agents talk to the API; the coordinator drives the cores and
rendezvous through the store) with real LLM competitors instead of scripted stubs.

Requires a running Arena server reachable at --server that shares this process's store backend:
  - LOCAL dry run (SQLite):  unset DATABASE_URL, start `python -m arena.cli serve`, then run this.
  - NEON (production):       DATABASE_URL set for BOTH the server and this process.

Examples:
  # local 1-game smoke test
  python tools/connected_sample.py --run-id sample_local --games 1 --server http://127.0.0.1:8000

  # a real k=10 run
  python tools/connected_sample.py --run-id sample_k10_1 --games 10 --server http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time

from persuasion_arena_agent.agent import ArenaAgent
from persuasion_arena_agent.credentials import CredentialsStore
from examples.session_agent import SessionAgent
from examples.file_memory_agent import FileMemoryAgent

from arena import store
from arena.connected import run_connected_batch
from arena.score import score_run

# Basic reference harnesses, alternated across seats so a run mixes independently-built harnesses
# (an LLM-session one and a write-events-to-a-file one). Each owns its own memory; the SDK only
# delivers the delta event stream.
HARNESSES = [SessionAgent, FileMemoryAgent]

# The five real competitors (name shown on the site -> OpenRouter model slug).
ROSTER = [
    ("GPT-5.4 mini",     "openai/gpt-5.4-mini"),
    ("Haiku 4.5",        "anthropic/claude-haiku-4.5"),
    ("Gemini 3.1 Flash", "google/gemini-3.1-flash-lite"),
    ("GPT-5.5",          "openai/gpt-5.5"),
    ("Sonnet 4.6",       "anthropic/claude-sonnet-4.6"),
]


def _log(msg: str) -> None:
    print(f"[sample] {msg}", flush=True)


def _run_agent(name: str, model: str, run_id: str, server: str, cred_path: str, harness_cls) -> None:
    """One agent identity: register -> sign up -> ready -> poll/act until the run completes.
    `harness_cls` is the participant's harness — it owns its memory; the SDK only delivers events."""
    try:
        agent = ArenaAgent(name=name, server=server, credentials=CredentialsStore(cred_path))
        ma = harness_cls(model)
        agent.on_event(ma.on_event)   # the harness folds each delta event into its OWN memory
        agent.act(ma.act)             # the harness decides from the memory it built
        signup = agent.signup(run_id=run_id)
        _log(f"{name} [{harness_cls.__name__}]: signed up ({signup.status}, seat={signup.seat})")
        agent.run_forever([signup])
        _log(f"{name}: done")
    except Exception as e:  # an agent thread dying must not wedge the others
        _log(f"{name}: ERROR {type(e).__name__}: {e}")


def _wait_for_active(run_id: str, need: int, timeout_s: float = 180.0) -> int:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        n = store.activate_run_if_ready(run_id)   # coordinator-driven, race-free
        if n >= need:
            return n
        time.sleep(0.5)
    return store.activate_run_if_ready(run_id)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a connected sample run with real model agents")
    p.add_argument("--run-id", required=True)
    p.add_argument("--server", default="http://127.0.0.1:8000")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--deck", default="arena")
    p.add_argument("--models", default=None,
                   help="comma-separated OpenRouter slugs to use instead of the default roster "
                        "(names derived from the slug); handy for cheap local validation")
    p.add_argument("--join", action="store_true",
                   help="agents-only: the run already exists and a REMOTE coordinator (e.g. the "
                        "per-run Modal container) drives it — don't create the run or coordinate "
                        "locally; just connect agents and wait for them to finish")
    p.add_argument("--wait", type=int, default=1200, help="join-mode: seconds to wait for agents")
    args = p.parse_args(argv)

    if args.models:
        roster = [(m.split("/")[-1], m) for m in (x.strip() for x in args.models.split(",")) if m]
    else:
        roster = list(ROSTER)
    players = len(roster)
    if args.join:
        _log(f"joining existing run {args.run_id} as agents only — a remote coordinator "
             f"(e.g. the per-run Modal container) drives the game")
    else:
        _log(f"creating open run {args.run_id} (onuw, players={players}, games={args.games}, "
             f"deck={args.deck}, seed={args.seed})")
        store.create_connected_run({
            "id": args.run_id,
            "game": "onuw",
            "label": "One Night Ultimate Werewolf",
            "status": "open",
            "n_games": args.games,
            "players": players,
            "seed_base": args.seed,
            "submitter": "connected-sample",
            "deck_preset": args.deck,
        })

    threads = []
    for i, (name, model) in enumerate(roster):
        harness_cls = HARNESSES[i % len(HARNESSES)]
        # A path that does NOT exist yet (an empty file would make CredentialsStore json.loads("") crash);
        # the dir exists so the SDK can write the credential on first register.
        cred_path = os.path.join(tempfile.mkdtemp(prefix="arena-cred-"), "cred.json")
        t = threading.Thread(target=_run_agent,
                             args=(name, model, args.run_id, args.server, cred_path, harness_cls),
                             daemon=True, name=name)
        t.start()
        threads.append(t)
        time.sleep(0.3)  # stagger registration so seat order is stable

    if args.join:
        _log("agents connected; the remote coordinator will seat them and run the game(s). "
             "waiting for agents to finish...")
        for t in threads:
            t.join(timeout=args.wait)
        alive = [t.name for t in threads if t.is_alive()]
        _log("all agents finished" if not alive else f"timed out; still running: {alive}")
        return 0 if not alive else 1

    _log("waiting for all agents to seat and ready...")
    active = _wait_for_active(args.run_id, players)
    if active < players:
        _log(f"only {active}/{players} agents became active — aborting (run left open)")
        return 1
    _log(f"all {active} agents active — coordinating {args.games} games")

    t0 = time.time()
    run_connected_batch(args.run_id, discussion_rounds=args.rounds)
    dt = time.time() - t0

    for t in threads:
        t.join(timeout=10)

    run = store.get_run(args.run_id)
    _log(f"run status={run['status']} games_saved={len(store.distinct_gids(args.run_id))} "
         f"team_split={run['team_split']} in {dt:.0f}s")
    scores = score_run(args.run_id)
    _log("scorecard (overall win% [95% CI] n, forfeits):")
    for name, d in sorted(scores.items(), key=lambda kv: -kv[1]["overall"]["rate"]):
        o = d["overall"]
        print(f"    {name:18} {int(o['rate']*100):3d}%  "
              f"[{int(o['lo']*100)}-{int(o['hi']*100)}%]  n={o['n']:<3} "
              f"forfeits={d['forfeits']}/{d['calls']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
