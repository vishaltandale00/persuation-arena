"""Run a connected ONUW run whose competitors are real coding-agent harnesses.

Five seats: three coding-agent brains (codex, the Claude Agent SDK / "Claude Code", and opencode,
each driving a persistent per-game session) plus two reference chat harnesses (an LLM-session and a
file-memory one on OpenRouter slugs) for contrast. Every harness owns its own memory; the SDK only
delivers the delta event stream — the same statefulness contract every participant plays under.

This is `connected_sample.py` with a mixed, coding-agent-heavy roster and per-seat harness selection.
A server must be reachable at --server and SHARE this process's store backend:

  LOCAL smoke (SQLite):  unset DATABASE_URL, start `python -m arena.cli serve`, then run this.
  NEON (production):     DATABASE_URL set for BOTH the server and this process (results show on the site).

Auth: codex uses its ChatGPT login (or OPENAI_API_KEY), the Claude Agent SDK uses your Claude
subscription via the `claude` CLI (or ANTHROPIC_API_KEY), opencode uses `opencode auth login`; the two
reference seats use OPENROUTER_API_KEY. Override a coding brain's model with ARENA_CODEX_MODEL /
ARENA_CLAUDE_AGENT_MODEL / ARENA_OPENCODE_MODEL (unset -> that tool's own default).

Examples:
  # cheap smoke — prove the coding agents connect and play one game end to end
  python tools/coding_agent_run.py --run-id smoke_coding --games 1 --rounds 2 --server http://127.0.0.1:8000

  # the real validation run (DATABASE_URL set -> visible on the live observer)
  python tools/coding_agent_run.py --run-id coding_k10 --games 10 --rounds 10 --server http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time

from persuasion_arena_agent.agent import ArenaAgent
from persuasion_arena_agent.credentials import CredentialsStore

from examples.codex_agent import CodexHarness
from examples.claude_agent_sdk_agent import ClaudeAgentSdkHarness
from examples.opencode_agent import OpencodeHarness
from examples.session_agent import SessionAgent
from examples.file_memory_agent import FileMemoryAgent

from arena import store
from arena.connected import run_connected_batch
from arena.score import score_run


# (display name -> a factory that builds a fresh harness instance). The three coding brains take their
# tool's default model (model=None); the two reference seats run named OpenRouter slugs. Factories so
# each seat's memory is isolated and nothing is shared across the run.
# opencode is omitted for now: it hangs on every turn (forfeits 3/3 even at a 240s deadline) — a
# default-model / non-interactive config issue, not auth. Re-add once ARENA_OPENCODE_MODEL is sorted.
# Ordered so the LAST seats are the ones reserved by --external N (host launches SEATS[:players-N]).
# Codex is last so a human participant can run the coding agent themselves while the host runs the rest.
SEATS = [
    ("Claude Code",    lambda: ClaudeAgentSdkHarness()),                            # Claude Agent SDK (coding)
    ("Session·GPT",    lambda: SessionAgent("openai/gpt-5.4-mini")),               # reference LLM-session
    ("FileMem·Haiku",  lambda: FileMemoryAgent("anthropic/claude-haiku-4.5")),     # reference file-memory
    ("Session·Gemini", lambda: SessionAgent("google/gemini-3.1-flash-lite")),      # reference LLM-session
    ("Codex",          lambda: CodexHarness()),                                     # codex exec (reserved last)
]


def _log(msg: str) -> None:
    print(f"[coding-run] {msg}", flush=True)


def _run_agent(name: str, make_harness, run_id: str, server: str, cred_path: str) -> None:
    """One agent identity: register -> sign up -> ready -> poll/act until the run completes.
    The harness owns its memory; the SDK only delivers the delta event stream."""
    try:
        agent = ArenaAgent(name=name, server=server, credentials=CredentialsStore(cred_path))
        h = make_harness()
        agent.on_event(h.on_event)   # fold each delta event into the harness's OWN memory
        agent.act(h.act)             # decide from the memory the harness built
        signup = agent.signup(run_id=run_id)
        _log(f"{name} [{type(h).__name__}]: signed up ({signup.status}, seat={signup.seat})")
        agent.run_forever([signup])
        _log(f"{name}: done")
    except Exception as e:  # one agent dying must not wedge the others
        _log(f"{name}: ERROR {type(e).__name__}: {e}")


def _wait_for_active(run_id: str, need: int, timeout_s: float) -> int:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        n = store.activate_run_if_ready(run_id)   # coordinator-driven, race-free
        if n >= need:
            return n
        time.sleep(0.5)
    return store.activate_run_if_ready(run_id)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Connected ONUW run with coding-agent harnesses")
    p.add_argument("--run-id", required=True)
    p.add_argument("--server", default="http://127.0.0.1:8000")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--deck", default="arena")
    p.add_argument("--ready-timeout", type=float, default=300.0,
                   help="seconds to wait for all coding agents to seat + ready (they can be slow to boot)")
    p.add_argument("--deadline", type=int, default=240,
                   help="per-turn reply deadline (s); MUST exceed a coding agent's think time or good "
                        "actions get defaulted as forfeits. The harness CLI wait is ARENA_AGENT_BRAIN_TIMEOUT "
                        "(~180s), so keep this above that.")
    p.add_argument("--external", type=int, default=0,
                   help="reserve N seats for externally-run agents: the host launches players-N agents "
                        "and coordinates; you run the other N yourself via `arena-agent play`. The last N "
                        "SEATS entries are the reserved ones.")
    args = p.parse_args(argv)

    players = len(SEATS)
    backend = "NEON" if os.environ.get("DATABASE_URL") else "SQLite"
    _log(f"creating open run {args.run_id} on {backend} (onuw, players={players}, games={args.games}, "
         f"rounds={args.rounds}, deck={args.deck}, seed={args.seed})")
    store.create_connected_run({
        "id": args.run_id,
        "game": "onuw",
        "label": "One Night Ultimate Werewolf",
        "status": "open",
        "n_games": args.games,
        "players": players,
        "seed_base": args.seed,
        "rounds": args.rounds,
        "submitter": "coding-agent-run",
        "deck_preset": args.deck,
    })

    mine = SEATS[: players - args.external] if args.external else SEATS
    if args.external:
        _log(f"reserving {args.external} of {players} seat(s) for EXTERNAL agents you run yourself.")
        _log(f"in another terminal (repo root), run each reserved agent, e.g. opencode pinned to GLM 5.2:")
        _log(f"  ARENA_OPENCODE_MODEL=openrouter/z-ai/glm-5.2 PYTHONPATH=. \\")
        _log(f"  .venv/bin/arena-agent play --run {args.run_id} --server {args.server} \\")
        _log(f"  --name 'GLM-opencode' examples/opencode_agent.py")

    threads = []
    for name, make_harness in mine:
        # A path that does NOT exist yet (an empty file makes CredentialsStore json.loads("") crash);
        # the dir exists so the SDK writes the credential on first register.
        cred_path = os.path.join(tempfile.mkdtemp(prefix="arena-cred-"), "cred.json")
        t = threading.Thread(target=_run_agent, args=(name, make_harness, args.run_id, args.server, cred_path),
                             daemon=True, name=name)
        t.start()
        threads.append(t)
        time.sleep(0.4)  # stagger registration so seat order is stable

    _log(f"launched {len(mine)} host agent(s); waiting for all {players} to seat and ready "
         f"({args.external} reserved for you) — start your agent(s) now...")
    active = _wait_for_active(args.run_id, players, args.ready_timeout)
    if active < players:
        _log(f"only {active}/{players} agents became active — aborting (run left open)")
        return 1
    _log(f"all {active} agents active — coordinating {args.games} games × {args.rounds} rounds "
         f"(turn deadline {args.deadline}s)")

    t0 = time.time()
    run_connected_batch(args.run_id, discussion_rounds=args.rounds, deadline_seconds=args.deadline)
    dt = time.time() - t0

    for t in threads:
        t.join(timeout=30)

    run = store.get_run(args.run_id)
    _log(f"run status={run['status']} games_saved={len(store.distinct_gids(args.run_id))} "
         f"team_split={run['team_split']} in {dt:.0f}s")
    scores = score_run(args.run_id)
    _log("scorecard (overall win% [95% CI] n, forfeits):")
    for name, d in sorted(scores.items(), key=lambda kv: -kv[1]["overall"]["rate"]):
        o = d["overall"]
        print(f"    {name:16} {int(o['rate']*100):3d}%  "
              f"[{int(o['lo']*100)}-{int(o['hi']*100)}%]  n={o['n']:<3} "
              f"forfeits={d['forfeits']}/{d['calls']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
