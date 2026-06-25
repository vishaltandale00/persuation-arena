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
import subprocess
import sys
import tempfile
import threading
import time

from persuasion_arena_agent.agent import ArenaAgent
from persuasion_arena_agent.credentials import CredentialsStore

from examples.codex_agent import CodexHarness
from examples.claude_agent_sdk_agent import ClaudeAgentSdkHarness
from examples.opencode_agent import OpencodeHarness
from examples.pi_agent import PiHarness
from examples.session_agent import SessionAgent
from examples.file_memory_agent import FileMemoryAgent

from arena import store
from arena.connected import run_connected_batch
from arena.score import score_run, score_runs
from arena.sharded import create_sharded_run, rollup_parent_status


# (display name -> a factory that builds a fresh harness instance). The three coding brains take their
# tool's default model (model=None); the two reference seats run named OpenRouter slugs. Factories so
# each seat's memory is isolated and nothing is shared across the run.
# Frontier-tier suite: similar-intelligence models across agentic harnesses. gpt-5.5 appears via BOTH
# codex and pi, isolating the harness effect at a fixed model. codex/claude-code are subsidized (Max/
# ChatGPT); the three pi seats route through paid OpenRouter. Reserved-last ordering for --external N.
SEATS = [
    ("ClaudeCode·Opus4.8", lambda: ClaudeAgentSdkHarness("claude-opus-4-8")),         # claude-code harness, Opus 4.8 (subsidized)
    ("Pi·Gemini3.5Flash",  lambda: PiHarness("openrouter/google/gemini-3.5-flash")),  # pi harness, Gemini 3.5 Flash
    ("Pi·GLM5.2",          lambda: PiHarness("openrouter/z-ai/glm-5.2")),             # pi harness, GLM 5.2 (OpenRouter)
    ("Pi·GPT5.5",          lambda: PiHarness("openrouter/openai/gpt-5.5")),           # pi harness, GPT-5.5 (harness control)
    ("Codex·GPT5.5",       lambda: CodexHarness("gpt-5.5")),                          # codex harness, GPT-5.5 (subsidized)
]


def _log(msg: str) -> None:
    print(f"[coding-run] {msg}", flush=True)


def _run_agent(name: str, make_harness, run_id: str, server: str, cred_path: str,
               join_token: str | None = None) -> None:
    """One agent identity: register -> sign up -> ready -> poll/act until the run completes.
    The harness owns its memory; the SDK only delivers the delta event stream.

    `join_token` is the per-parent shard secret (INV-4); a shard host passes its child's token so the
    signup is accepted. None for normal runs (token omitted from the request body, INV-2)."""
    try:
        agent = ArenaAgent(name=name, server=server, credentials=CredentialsStore(cred_path))
        h = make_harness()
        agent.on_event(h.on_event)   # fold each delta event into the harness's OWN memory
        agent.act(h.act)             # decide from the memory the harness built
        signup = agent.signup(run_id=run_id, join_token=join_token)
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


# --- sharded launcher (SPEC-parallel-shards.md §6.9(b), D9/D10) ----------------------------------
#
# K>1 fans the logical run into a parent + K child shards. Credential-sharing across an identity's K
# shards is FILE-BASED (D9): the launcher pre-registers each identity ONCE into a cred file, then
# every shard subprocess reads the same file -> the same agent_id, so one identity is one rating
# competitor across all shards. Startup is independent (no cross-shard barrier, D10); the launcher
# block-monitors `rollup_parent_status` as a non-load-bearing convenience.


def _pre_register_creds(creds_dir: str, names: list[str], server: str) -> dict[str, str]:
    """Pre-register each identity ONCE into its own cred file under creds_dir (shared across shards).

    Returns {display_name: cred_path}. Registration is idempotent per the SDK (it writes the cred on
    first register and reuses it thereafter), so a re-run with the same creds_dir keeps the agent_ids.
    """
    os.makedirs(creds_dir, exist_ok=True)
    paths: dict[str, str] = {}
    for idx, name in enumerate(names):
        cred_path = os.path.join(creds_dir, f"identity_{idx}.json")
        # register ONCE so the cred file exists before any shard reads it; the SDK registers on
        # first use and persists the agent_id to this file (idempotent thereafter).
        ArenaAgent(name=name, server=server, credentials=CredentialsStore(cred_path)).ensure_registered()
        paths[name] = cred_path
    return paths


def _spawn_shard_hosts(child_ids: list[str], creds_dir: str, args) -> list[subprocess.Popen]:
    """Spawn one single-shard host subprocess per child (process per shard, D9). Each re-invokes THIS
    module in --shard-child mode pointed at one child run_id, reading the shared cred files."""
    procs: list[subprocess.Popen] = []
    for cid in child_ids:
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--shard-child", cid, "--creds-dir", creds_dir,
            "--run-id", cid, "--server", args.server,
            "--games", str(args.games), "--rounds", str(args.rounds),
            "--seed", str(args.seed), "--deck", args.deck,
            "--ready-timeout", str(args.ready_timeout), "--deadline", str(args.deadline),
        ]
        procs.append(subprocess.Popen(cmd))
        time.sleep(0.4)  # stagger so each shard's seat order is stable
    return procs


def _monitor_shards(parent_id: str, child_ids: list[str], procs: list[subprocess.Popen],
                    timeout_s: float) -> str:
    """Block-monitor the shards (non-load-bearing, D10): poll rollup_parent_status and per-shard
    health until the parent rolls up to a terminal state or all hosts exit. Returns the final rollup.

    A shard host can exit BEFORE its child run reaches a terminal state — e.g. `_run_shard_child`
    returns nonzero because not enough agents activated, leaving the child 'waiting'/'ready_required'.
    Once that host process is gone, nothing will ever advance that child, so the rollup would sit at
    'running' forever and a naive wait would burn the whole games*rounds*deadline budget. So when ALL
    host subprocesses have exited we stop waiting immediately: if the rollup is still non-terminal we
    force the unfinished child(ren) to 'partial' (monotonic in the store, so a child that actually
    reached 'done' is left untouched) and return the resolved rollup. The happy path — every host
    exits cleanly and the children reach 'done' — still returns 'done'."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = rollup_parent_status(parent_id)
        all_exited = all(p.poll() is not None for p in procs)
        if status in {"done", "partial"} and all_exited:
            return status
        if all_exited:
            # Every host is dead but the rollup is non-terminal -> some child is stranded
            # (no live process can ever finish it). Resolve it now instead of hanging.
            return _resolve_dead_shards(parent_id, child_ids)
        time.sleep(1.0)
    # Timed out with at least one host still running: terminate the stragglers, then resolve.
    for p in procs:
        if p.poll() is None:
            p.terminate()
    status = rollup_parent_status(parent_id)
    if status in {"done", "partial"}:
        return status
    return _resolve_dead_shards(parent_id, child_ids)


def _resolve_dead_shards(parent_id: str, child_ids: list[str]) -> str:
    """Mark every not-yet-terminal child 'partial' so the parent rolls up to a terminal state.

    `update_run_status` is monotonic (it never overwrites a 'done'), so a child that truly finished
    keeps 'done'; only the stranded ones flip to 'partial'. Returns the post-resolution rollup."""
    for cid in child_ids:
        run = store.get_run(cid) or {}
        if run.get("status") not in {"done", "partial"}:
            store.update_run_status(cid, "partial")
    return rollup_parent_status(parent_id)


def _run_sharded(args) -> int:
    """K>1: create parent+children, pre-register shared creds, spawn K shard hosts, block-monitor."""
    players = len(SEATS)
    backend = "NEON" if os.environ.get("DATABASE_URL") else "SQLite"
    _log(f"creating SHARDED run {args.run_id} on {backend} (onuw, players={players}, "
         f"games={args.games}, shards={args.shards}, rounds={args.rounds}, seed={args.seed})")
    parent_cfg = {
        "id": args.run_id,
        "game": "onuw",
        "label": "One Night Ultimate Werewolf",
        "status": "open",
        "n_games": args.games,  # GLOBAL N (D8); each child slices to its shard
        "players": players,
        "seed_base": args.seed,
        "submitter": "coding-agent-run",
        "deck_preset": args.deck,
    }
    child_ids = create_sharded_run(parent_cfg, args.shards)
    _log(f"parent {args.run_id} -> {len(child_ids)} child shard(s): {', '.join(child_ids)}")

    creds_dir = args.creds_dir or tempfile.mkdtemp(prefix=f"arena-shards-{args.run_id}-")
    names = [name for name, _ in SEATS]
    _pre_register_creds(creds_dir, names, args.server)
    _log(f"pre-registered {len(names)} shared identities into {creds_dir}")

    procs = _spawn_shard_hosts(child_ids, creds_dir, args)
    _log(f"spawned {len(procs)} shard host process(es); block-monitoring rollup ...")
    monitor_timeout = max(args.ready_timeout, 1.0) + args.games * args.rounds * args.deadline
    status = _monitor_shards(args.run_id, child_ids, procs, monitor_timeout)
    _log(f"parent rollup status = {status}")

    scores = score_runs(child_ids)
    _log("aggregated scorecard (overall win% [95% CI] n, forfeits):")
    for name, d in sorted(scores.items(), key=lambda kv: -kv[1]["overall"]["rate"]):
        o = d["overall"]
        print(f"    {name:16} {int(o['rate']*100):3d}%  "
              f"[{int(o['lo']*100)}-{int(o['hi']*100)}%]  n={o['n']:<3} "
              f"forfeits={d['forfeits']}/{d['calls']}", flush=True)
    return 0 if status == "done" else 1


def _run_shard_child(args) -> int:
    """INTERNAL single-shard host: sign this process's agents into ONE child run and coordinate it.

    Each agent reads its SHARED cred file from --creds-dir (same agent_id across shards). The child
    run already exists (created by the launcher via create_sharded_run); this process only seats +
    coordinates it. Process isolation (D9) means this host holds signups for exactly one run."""
    child_id = args.shard_child
    creds_dir = args.creds_dir
    players = len(SEATS)
    # The child row carries the per-parent join_token (minted by create_sharded_run); a stray agent
    # cannot guess it, so passing it here is what makes the child joinable for this host (INV-4).
    child_run = store.get_run(child_id) or {}
    join_token = child_run.get("join_token")
    _log(f"[shard host] coordinating child {child_id} (players={players})")

    threads = []
    for idx, (name, make_harness) in enumerate(SEATS):
        cred_path = os.path.join(creds_dir, f"identity_{idx}.json")
        t = threading.Thread(target=_run_agent,
                             args=(name, make_harness, child_id, args.server, cred_path, join_token),
                             daemon=True, name=f"{child_id}:{name}")
        t.start()
        threads.append(t)
        time.sleep(0.4)

    active = _wait_for_active(child_id, players, args.ready_timeout)
    if active < players:
        _log(f"[shard host] only {active}/{players} agents active for {child_id} — aborting")
        return 1
    run_connected_batch(child_id, discussion_rounds=args.rounds, deadline_seconds=args.deadline)
    for t in threads:
        t.join(timeout=30)
    run = store.get_run(child_id)
    _log(f"[shard host] {child_id} status={run['status']} "
         f"games_saved={len(store.distinct_gids(child_id))}")
    return 0 if run["status"] == "done" else 1


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
    p.add_argument("--shards", type=int, default=1,
                   help="split the run into K concurrent child shards (SPEC-parallel-shards.md, D3; "
                        "default 1 = today's single run, hard cap from ARENA_MAX_SHARDS). With K>1 the "
                        "launcher creates a parent + K children, pre-registers the N identities ONCE "
                        "each to shared cred files, spawns K single-shard host subprocesses, then "
                        "block-monitors rollup_parent_status and prints the aggregated scorecard.")
    p.add_argument("--shard-child", default=None,
                   help="INTERNAL single-shard host mode: sign this process's agents into ONE child "
                        "run_id and run it to completion (spawned by the launcher per shard).")
    p.add_argument("--creds-dir", default=None,
                   help="INTERNAL: directory of pre-registered per-identity cred files shared across "
                        "shards (shared creds => one rating competitor per identity across all K).")
    args = p.parse_args(argv)

    # INTERNAL single-shard host mode: this process owns exactly one child run (process isolation,
    # D9). It reads the pre-registered cred files (shared across shards => same agent_id), signs its
    # agents into that one child, and coordinates it to completion.
    if args.shard_child:
        return _run_shard_child(args)

    if args.shards and args.shards > 1:
        return _run_sharded(args)

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
