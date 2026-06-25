"""Orchestrate a connected WolfForgeAgentV2 evaluation run end to end.

Seats one WolfForgeV2 agent + one frozen CharismaBaseline agent + a documented opponent pool into a
single ALL-CONNECTED ONUW run (the current connected tooling drives every seat through the same
coordinator; it does not mix connected and in-process static agents in one run). Both the V2 agent
and the baseline reuse the IDENTICAL connected harness machinery and differ only in the strategy
prompt — the single experimental variable, mirroring the original WolfForge ablation.

This is the orchestrator the evaluation commands in WOLFFORGE_V2_EVAL.md call. It mirrors
tools/connected_sample.py (create open run -> connect agents -> wait active -> coordinate -> score).

--fake-brain makes the WHOLE run model-free and FREE: V2 and the baseline use a deterministic
no-model brain (every turn resolves through the legal deterministic fallback) and opponents use the
reference random agent. That is the no-cost Stage-0 local connected protocol smoke.

Local (SQLite) usage:
    # terminal 1: start the server sharing this process's SQLite store
    python -m arena.cli serve --port 8000
    # terminal 2: model-free protocol smoke (no OpenRouter calls, no cost)
    python tools/wolfforge_v2_run.py --run-id wf_v2_smoke --games 5 --fake-brain \
        --server http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time

# Allow running as a plain script (`python tools/wolfforge_v2_run.py`): put the repo root on the path
# so the top-level `examples` package imports even when it is not installed into the environment.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persuasion_arena_agent.agent import ArenaAgent
from persuasion_arena_agent.credentials import CredentialsStore

from arena import store
from arena.connected import run_connected_batch
from arena.score import score_run

from examples.wolfforge_v2_agent import BrainResult, V2Config, WolfForgeV2Agent
from examples import random_agent
from examples.session_agent import SessionAgent

DEFAULT_MODEL = "openai/gpt-4o-mini"
# Stable, documented opponent pool: three reference model-backed seats. Cheap enough to iterate,
# varied enough to expose strategy differences. Override with --opponent-models.
DEFAULT_OPPONENTS = [
    ("Opp1", "openai/gpt-4o-mini"),
    ("Opp2", "openai/gpt-4o-mini"),
    ("Opp3", "openai/gpt-4o-mini"),
]


class _NoModelBrain:
    """Deterministic, model-free brain: always 'fails' so the harness takes its legal fallback.
    Used by --fake-brain to exercise the connected protocol with zero API cost."""

    def complete(self, messages, *, response_format, timeout) -> BrainResult:
        return BrainResult(content="", ok=False, error_type="fake_brain_no_model")


def _log(msg: str) -> None:
    print(f"[wf-v2-run] {msg}", flush=True)


def _cred_path() -> str:
    return os.path.join(tempfile.mkdtemp(prefix="arena-cred-"), "cred.json")


def _run_v2_agent(name: str, run_id: str, server: str, baseline: bool, fake: bool,
                  model: str) -> None:
    try:
        cfg = V2Config.from_env()
        cfg.model = model
        brain = _NoModelBrain() if fake else None
        if baseline:
            harness = WolfForgeV2Agent.charisma_baseline(run_id=run_id, config=cfg, brain=brain,
                                                         agent_name=name)
        else:
            harness = WolfForgeV2Agent(run_id=run_id, config=cfg, brain=brain, agent_name=name)
        agent = ArenaAgent(name=name, server=server, credentials=CredentialsStore(_cred_path()),
                           model=model, harness=harness.harness)
        agent.on_event(harness.on_event)
        agent.act(harness.act)
        signup = agent.signup(run_id=run_id)
        _log(f"{name}: signed up ({signup.status}, seat={signup.seat})")
        agent.run_forever([signup])
    except Exception as e:  # one agent thread dying must not wedge the others
        _log(f"{name}: ERROR {type(e).__name__}: {e}")


def _run_opponent(name: str, model: str, run_id: str, server: str, fake: bool) -> None:
    try:
        agent = ArenaAgent(name=name, server=server, credentials=CredentialsStore(_cred_path()),
                           model=model, harness="reference")
        if fake:
            agent.act(random_agent.act)  # model-free, legal, blind reference play
        else:
            ma = SessionAgent(model)
            agent.on_event(ma.on_event)
            agent.act(ma.act)
        signup = agent.signup(run_id=run_id)
        _log(f"{name}: signed up ({signup.status}, seat={signup.seat})")
        agent.run_forever([signup])
    except Exception as e:
        _log(f"{name}: ERROR {type(e).__name__}: {e}")


def _wait_for_active(run_id: str, need: int, timeout_s: float = 180.0) -> int:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        n = store.activate_run_if_ready(run_id)
        if n >= need:
            return n
        time.sleep(0.5)
    return store.activate_run_if_ready(run_id)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Connected WolfForgeAgentV2 vs CharismaBaseline run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--server", default="http://127.0.0.1:8000")
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--seed", type=int, default=70000)
    p.add_argument("--deck", default="arena")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model for V2 + baseline (env WOLFFORGE_V2_MODEL also applies)")
    p.add_argument("--opponent-models", default=None,
                   help="comma-separated opponent OpenRouter slugs (default: 3x gpt-4o-mini)")
    p.add_argument("--fake-brain", action="store_true",
                   help="model-free: no OpenRouter calls, deterministic legal play (no cost)")
    p.add_argument("--join", action="store_true",
                   help="run already exists and a remote coordinator drives it; just connect agents")
    p.add_argument("--wait", type=int, default=1200)
    args = p.parse_args(argv)

    # Diagnostic transparency: the coordinator/runner and the --server process MUST share one store
    # backend, or signups/turns created by one are invisible to the other. This line names the active
    # backend safely (never the Postgres URL) so a mismatch is obvious. See WOLFFORGE_AGENT_V2.md.
    _log(f"active store: {store.active_backend_label()} "
         f"(the --server process must use the SAME backend)")

    if args.opponent_models:
        opponents = [(f"Opp{i+1}", m) for i, m in
                     enumerate(x.strip() for x in args.opponent_models.split(",") if x.strip())]
    else:
        opponents = list(DEFAULT_OPPONENTS)
    players = 2 + len(opponents)

    if not args.join:
        _log(f"creating open run {args.run_id} (onuw, players={players}, games={args.games}, "
             f"deck={args.deck}, seed={args.seed}, fake_brain={args.fake_brain})")
        store.create_connected_run({
            "id": args.run_id, "game": "onuw", "label": "WolfForgeAgentV2 evaluation",
            "status": "open", "n_games": args.games, "players": players, "seed_base": args.seed,
            "submitter": "wolfforge-v2-run", "deck_preset": args.deck,
        })

    threads: list[threading.Thread] = []

    def spawn(target, targs, name):
        t = threading.Thread(target=target, args=targs, daemon=True, name=name)
        t.start()
        threads.append(t)
        time.sleep(0.3)  # stagger registration so seat order is stable

    spawn(_run_v2_agent, ("WolfForgeV2", args.run_id, args.server, False, args.fake_brain, args.model),
          "WolfForgeV2")
    spawn(_run_v2_agent, ("CharismaBaseline", args.run_id, args.server, True, args.fake_brain, args.model),
          "CharismaBaseline")
    for name, model in opponents:
        spawn(_run_opponent, (name, model, args.run_id, args.server, args.fake_brain), name)

    if args.join:
        _log("agents connected; remote coordinator drives the games. waiting for agents to finish...")
        for t in threads:
            t.join(timeout=args.wait)
        return 0

    _log("waiting for all agents to seat and ready...")
    active = _wait_for_active(args.run_id, players)
    if active < players:
        _log(f"only {active}/{players} agents active — aborting (run left open)")
        return 1
    _log(f"all {active} agents active — coordinating {args.games} games (workers=1)")

    t0 = time.time()
    run_connected_batch(args.run_id, discussion_rounds=args.rounds)
    dt = time.time() - t0
    for t in threads:
        t.join(timeout=10)

    run = store.get_run(args.run_id)
    _log(f"run status={run['status']} games={len(store.distinct_gids(args.run_id))} in {dt:.0f}s")
    scores = score_run(args.run_id)
    for name, d in sorted(scores.items(), key=lambda kv: -kv[1]["overall"]["rate"]):
        o = d["overall"]
        print(f"    {name:18} {int(o['rate']*100):3d}%  [{int(o['lo']*100)}-{int(o['hi']*100)}%]  "
              f"n={o['n']:<3} forfeits={d['forfeits']}/{d['calls']}", flush=True)
    _log("NOTE: use tools/wolfforge_v2_eval.py for the paired V2-vs-baseline statistics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
