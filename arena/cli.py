"""Persuasion Arena CLI — the drivable surface.

  python -m arena.cli run --game onuw --games 20            # play a run
  python -m arena.cli run --game avalon --games 6 --workers 8
  python -m arena.cli score --run run_9000                 # print the leaderboard
  python -m arena.cli runs                                  # list runs
  python -m arena.cli agents                                # show the roster
  python -m arena.cli worker --server https://... --owner alice
  python -m arena.cli serve --port 8000                     # start the observer server
"""
from __future__ import annotations

import argparse
import os
import socket
import threading
import time

from .config import REASONING_EFFORTS, SETTINGS, caps_with_overrides


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _run_caps_from_args(args):
    return caps_with_overrides(
        reasoning_effort=args.reasoning_effort,
        max_tokens_per_turn=args.max_tokens_per_turn,
        temperature=args.temperature,
        retries=args.retries,
        prior_message_turns=args.prior_message_turns,
        discussion_rounds=args.rounds,
    )


def _run_caps_from_job(job: dict):
    fields = {}
    for agent in job.get("agents") or []:
        for key in ("reasoning_effort", "max_tokens_per_turn", "temperature", "retries",
                    "prior_message_turns"):
            if key in agent and agent[key] is not None and key not in fields:
                fields[key] = agent[key]
        if "max_tokens" in agent and "max_tokens_per_turn" not in fields:
            fields["max_tokens_per_turn"] = agent["max_tokens"]
        if len(fields) >= 5:
            break
    if "max_tokens_per_turn" in fields:
        fields["max_tokens_per_turn"] = int(fields["max_tokens_per_turn"])
    if "temperature" in fields:
        fields["temperature"] = float(fields["temperature"])
    if "retries" in fields:
        fields["retries"] = int(fields["retries"])
    if "prior_message_turns" in fields:
        fields["prior_message_turns"] = int(fields["prior_message_turns"])
    return caps_with_overrides(**fields, discussion_rounds=int(job["rounds"]))


def _use_local_store() -> None:
    """Pin the LOCAL CLI commands (run/score/runs) to the local SQLite store by dropping
    DATABASE_URL (which .env / the environment may set). These never touch the remote DB — the
    only path to the prod leaderboard is `arena push`, which uses the Vercel JS API. Remote-by-design
    commands (worker / serve / connected) are intentionally NOT routed through this."""
    os.environ.pop("DATABASE_URL", None)


def _run(args):
    from .batch import run_batch
    _use_local_store()
    rid = args.run_id or f"run_{args.seed}"
    caps = _run_caps_from_args(args)
    run_batch(game=args.game, n_games=args.games, seed_base=args.seed, run_id=rid,
              workers=args.workers, discussion_rounds=args.rounds, deck_preset=args.deck,
              caps=caps, deal_schedule=args.deal_schedule)
    print(f"\nDone. View at http://localhost:{args.port}/observer.html  (run: {rid})")
    print(f"Score with: python -m arena.cli score --run {rid}")


def _score(args):
    from .score import score_run
    from . import store
    _use_local_store()
    run = store.get_run(args.run)
    if run is None:
        print("run not found")
        return
    have = store.distinct_gids(args.run)
    if len(have) < run["n_games"] and not args.allow_partial:
        print(f"run '{args.run}' is incomplete: {len(have)}/{run['n_games']} games saved "
              f"(status={run['status']}). Re-run the missing games, or pass --allow-partial "
              f"to score the partial run anyway (sample size may be skewed).")
        return
    sc = score_run(args.run)
    if not sc:
        print("no scores (no games recorded for this run)")
        return
    if len(have) < run["n_games"]:
        print(f"WARNING: scoring a PARTIAL run ({len(have)}/{run['n_games']} games) — "
              f"sample size may be skewed.\n")
    rows = sorted(sc.items(), key=lambda kv: -kv[1]["overall"]["rate"])
    print(f"{'agent':10} {'win%':>5}  {'95% CI':>12}  {'n':>3}  {'fft%':>4}  by faction")
    for name, d in rows:
        o, g, e = d["overall"], d["good"], d["evil"]
        ci = f"{int(o['lo']*100)}-{int(o['hi']*100)}%"
        fft = int(round(d["forfeit_rate"] * 100))
        print(f"{name:10} {int(o['rate']*100):4d}%  {ci:>12}  {o['n']:>3}  {fft:>3}%  "
              f"good {int(g['rate']*100)}% (n{g['n']}) / evil {int(e['rate']*100)}% (n{e['n']})")
        roles = "  ".join(f"{r} {int(c['rate']*100)}% (n{c['n']})" for r, c in d["by_role"].items())
        print(f"{'':10} roles: {roles}")


def _runs(args):
    from . import store
    _use_local_store()
    for r in store.list_runs():
        print(f"{r['id']:12} {r['game']:7} {r['status']:8} {r['n_games']:>3} games  "
              f"split {r['team_split']}  {r['created']}")


def _agents(args):
    for s in SETTINGS.roster():
        print(f"{s.name:10} {s.model:42} harness={s.harness}")


def _serve(args):
    import uvicorn
    uvicorn.run("arena.server:app", host="127.0.0.1", port=args.port)


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _worker_post(server: str, path: str, token: str | None, body: dict) -> dict:
    import httpx
    r = httpx.post(server + path, headers=_headers(token), json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def _worker_get(server: str, path: str, token: str | None = None) -> dict:
    import httpx
    r = httpx.get(server + path, headers=_headers(token), timeout=30)
    r.raise_for_status()
    return r.json()


def _run_claimed_job(args, job: dict, worker_id: str, token: str | None) -> None:
    from .batch import run_batch
    from .config import AgentSpec
    from . import store

    server = args.server.rstrip("/")
    lease = args.lease
    stop = threading.Event()

    def heartbeat_loop():
        interval = max(5, lease // 3)
        while not stop.wait(interval):
            try:
                _worker_post(server, "/api/jobs/heartbeat", token, {
                    "owner": job["owner"], "job_id": job["id"], "worker_id": worker_id,
                    "lease_seconds": lease,
                })
            except Exception as e:
                print(f"[worker {worker_id}] heartbeat failed: {type(e).__name__}: {e}", flush=True)

    beat = threading.Thread(target=heartbeat_loop, daemon=True)
    beat.start()

    try:
        try:
            current = _worker_get(server, f"/api/runs/{job['run_id']}")
            existing = {int(g["gid"]) for g in current.get("games", [])}
        except Exception:
            existing = set()

        roster = [AgentSpec(name=a["name"], model=a["model"], harness=a.get("harness", "base"))
                  for a in job["agents"]]
        caps = _run_caps_from_job(job)

        def publish(gid: int, transcript: dict, agents: list[dict]) -> None:
            _worker_post(server, "/api/ingest", token, {
                "owner": job["owner"], "job_id": job["id"], "run_id": job["run_id"],
                "gid": gid, "transcript": transcript, "agents": agents,
            })

        print(f"[worker {worker_id}] claimed {job['id']} run={job['run_id']} "
              f"owner={job['owner']} existing={sorted(existing)}", flush=True)
        run_batch(game=job["game"], n_games=int(job["n_games"]), seed_base=int(job["seed_base"]),
                  run_id=job["run_id"], roster=roster, workers=args.workers,
                  discussion_rounds=int(job["rounds"]), deck_preset=job.get("deck_preset"),
                  skip_gids=existing, caps=caps,
                  on_game_saved=publish)
        local = store.get_run(job["run_id"])
        status = (local or {}).get("status", "done")
        if status not in {"done", "partial"}:
            status = "done"
        _worker_post(server, "/api/jobs/complete", token, {
            "owner": job["owner"], "job_id": job["id"], "worker_id": worker_id,
            "status": status,
        })
        print(f"[worker {worker_id}] completed {job['id']} status={status}", flush=True)
    except Exception as e:
        _worker_post(server, "/api/jobs/complete", token, {
            "owner": job["owner"], "job_id": job["id"], "worker_id": worker_id,
            "status": "failed", "error": f"{type(e).__name__}: {e}",
        })
        raise
    finally:
        stop.set()


def _worker(args):
    # The worker must run games locally and publish via HTTP. If DATABASE_URL is present
    # from a Vercel/Neon env pull, do not let store.py write directly to the central DB.
    os.environ.pop("DATABASE_URL", None)
    args.server = (args.server or os.environ.get("ARENA_SERVER_URL") or "http://127.0.0.1:8000").rstrip("/")
    args.owner = args.owner or os.environ.get("ARENA_WORKER_OWNER") or os.getlogin()
    token = args.token or os.environ.get("ARENA_WORKER_TOKEN") or os.environ.get("INGEST_TOKEN")
    worker_id = args.worker_id or f"{socket.gethostname()}-{os.getpid()}"

    print(f"[worker {worker_id}] polling {args.server} as owner={args.owner}", flush=True)
    while True:
        claim = _worker_post(args.server, "/api/jobs/claim", token, {
            "owner": args.owner, "worker_id": worker_id, "lease_seconds": args.lease,
        })
        job = claim.get("job")
        if not job:
            if args.once:
                print(f"[worker {worker_id}] no job available", flush=True)
                return
            time.sleep(args.poll)
            continue
        _run_claimed_job(args, job, worker_id, token)
        if args.once:
            return


# --- push: upload a finished LOCAL run to the prod Neon/Vercel leaderboard -------------------
# Disaster-protection only (not anti-cheat). The pure helpers below are factored out so they can be
# unit-tested offline without a live server or prod DB.

# Canonical production host used when neither --server nor $ARENA_SERVER_URL is set.
# NOTE: the Vercel project is 'persuation-arena' (the codebase's historical misspelling), so the
# live host is persuation-arena.vercel.app — NOT 'persuasion-arena'.
PROD_SERVER_URL = "https://persuation-arena.vercel.app"

PUSHABLE_RUN_STATUSES = {"done", "partial"}


def push_token_or_register(server: str, display_name: str | None = None) -> str:
    """Resolve the bearer token push sends to /api/runs/import — fully self-service, no hand-minting.

    Resolution order, keyed by `server`:
      1. $ARENA_INGEST_TOKEN, if set (explicit override / CI secret),
      2. a cached `pa_live_` token in CredentialsStore for this server,
      3. otherwise POST /api/agents/register (via client.register_agent) to mint a fresh identity,
         cache the returned `pa_live_` token in CredentialsStore keyed by server, and use it.

    Prints which identity is in use with the token redacted, and returns the raw token."""
    from persuasion_arena_agent.credentials import CredentialsStore, redact_token

    server = server.rstrip("/")

    env_token = (os.environ.get("ARENA_INGEST_TOKEN") or "").strip()
    if env_token:
        print(f"push: identity from $ARENA_INGEST_TOKEN (token {redact_token(env_token)})")
        return env_token

    store = CredentialsStore()
    cached = store.get(server)
    if cached and cached.agent_token:
        print(f"push: identity '{cached.display_name}' (agent_id={cached.agent_id}, "
              f"token {redact_token(cached.agent_token)}) [cached]")
        return cached.agent_token

    from persuasion_arena_agent.client import ArenaHttpClient

    name = display_name or socket.gethostname()
    print(f"push: no cached identity for {server} — registering as '{name}'...")
    client = ArenaHttpClient(server=server)
    try:
        creds = client.register_agent(name)
    finally:
        client.close()
    store.save(creds)
    print(f"push: registered identity '{creds.display_name}' (agent_id={creds.agent_id}, "
          f"token {redact_token(creds.agent_token)}) [cached -> {store.path}]")
    return creds.agent_token


def gid_diff(local_gids, remote_run_json: dict | None, force: bool = False) -> list[int]:
    """The gids to upload: local gids not already on the board (or all local gids when --force).
    remote_run_json is the GET /api/runs/{id} body (key 'games':[{gid,...}]); None/404 -> upload all.
    Mirrors the worker's resume diff (cli.py _run_claimed_job)."""
    local = sorted({int(g) for g in local_gids})
    if force:
        return local
    remote = set()
    if remote_run_json:
        for g in remote_run_json.get("games", []) or []:
            try:
                remote.add(int(g["gid"]))
            except (KeyError, TypeError, ValueError):
                continue
    return [g for g in local if g not in remote]


def build_run_header(run: dict) -> dict:
    """The run header in the import payload — exactly the keys store.save_run / the runs upsert need
    (agents stays a list; the server JSON.stringifies it into agents_json)."""
    return {
        "id": run["id"], "game": run["game"], "label": run["label"], "status": run["status"],
        "n_games": run["n_games"], "players": run["players"], "seed_base": run["seed_base"],
        "created": run.get("created"), "created_utc": run.get("created_utc"),
        "submitter": run.get("submitter"), "deck_preset": run.get("deck_preset"),
        "agents": run.get("agents", []),
    }


def build_import_payload(run: dict, gids, get_game) -> dict:
    """Assemble the batched {run, games:[{gid, transcript, agents}]} payload for /api/runs/import.

    The server derives game_players FROM transcript['players'] joined to agents[] by seat — exactly
    like store.save_game — so we send the full transcript plus the run roster (agents) per game and
    do NOT flatten a per-seat players array. `get_game(run_id, gid)` returns the stored transcript."""
    agents = run.get("agents", [])
    games = []
    for gid in gids:
        transcript = get_game(run["id"], int(gid))
        if transcript is None:
            continue
        games.append({"gid": int(gid), "transcript": transcript, "agents": agents})
    return {"run": build_run_header(run), "games": games}


def _push_one(args, run_id: str, token: str | None) -> dict:
    """Upload the missing games of a single local run. Returns a small summary dict."""
    from . import store
    server = args.server
    local_gids = store.distinct_gids(run_id)
    run = store.get_run(run_id)
    try:
        remote = _worker_get(server, f"/api/runs/{run_id}", token)
    except Exception:
        remote = None  # 404 / unreachable detail -> treat as empty board (upload all)
    missing = gid_diff(local_gids, remote, force=args.force)
    summary = {"run_id": run_id, "status": run["status"], "local": len(local_gids),
               "missing": len(missing), "uploaded": 0}
    if not missing:
        print(f"  {run_id}: up to date ({len(local_gids)} games on board)")
        return summary
    payload = build_import_payload(run, missing, store.get_game)
    if args.dry_run:
        print(f"  {run_id}: DRY-RUN would upload {len(payload['games'])} game(s): {missing}")
        return summary
    resp = _worker_post(server, "/api/runs/import", token, payload)
    summary["uploaded"] = len(payload["games"])
    print(f"  {run_id}: uploaded {len(payload['games'])} game(s) {missing} -> {resp}")
    return summary


def _push(args):
    """Upload the missing games of one (or --all) local run(s) to the prod leaderboard.

    No Neon/DATABASE_URL creds required: this talks to the prod HTTP API only. The server
    recomputes ratings on write (api/runs/import compute-on-write), so there is NO local recompute
    phase and no prod DB access from the client. Returns the manifest (list of per-run summaries)."""
    from . import store

    # Validate args BEFORE resolving/minting a token, so a bare `arena push` usage error never
    # triggers an auto-registration.
    if not args.all and not args.run:
        print("ERROR: pass --run RUN_ID (or --all to push every done/partial local run)")
        raise SystemExit(2)

    # POP DATABASE_URL so the READ phase (store.get_run / get_game / distinct_gids) hits the LOCAL
    # SQLite, not a prod Neon URL the env may carry. The client never connects to prod's DB.
    os.environ.pop("DATABASE_URL", None)
    args.server = (args.server or os.environ.get("ARENA_SERVER_URL") or PROD_SERVER_URL).rstrip("/")
    token = push_token_or_register(args.server, getattr(args, "as_name", None))

    if args.all:
        run_ids = [r["id"] for r in store.list_runs() if r["status"] in PUSHABLE_RUN_STATUSES]
        if not run_ids:
            print("no local runs with status done/partial to push")
            return []
    else:
        run_ids = [args.run]

    print(f"push: server={args.server} runs={run_ids} "
          f"{'(dry-run)' if args.dry_run else ''}{' (force)' if args.force else ''}".rstrip())
    summaries = []
    for run_id in run_ids:
        run = store.get_run(run_id)
        if run is None:
            print(f"  {run_id}: NOT FOUND in local store — skipping")
            continue
        summaries.append(_push_one(args, run_id, token))

    total_uploaded = sum(s["uploaded"] for s in summaries)
    print(f"push: {total_uploaded} game(s) uploaded across {len(summaries)} run(s)")
    if args.dry_run:
        print("push: dry-run — nothing written")
    elif total_uploaded:
        print("push: done. The leaderboard updates server-side (compute-on-write); no local recompute.")
    return summaries


def main():
    p = argparse.ArgumentParser(prog="arena")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="play a run of N games")
    r.add_argument("--game", default="onuw", choices=["onuw", "avalon"])
    r.add_argument("--games", type=int, default=20)
    r.add_argument("--seed", type=int, default=9000)
    r.add_argument("--run-id", dest="run_id", default=None)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--rounds", type=_positive_int, default=20,
                   help="discussion budget: max ONUW messages; round cap for other games")
    r.add_argument("--deck", default="arena", help="ONUW deck preset: arena, classic, or tanner")
    r.add_argument("--deal-schedule", choices=["random", "balanced"], default=None,
                   help="initial deal scheduling; default is balanced for ONUW and random for other games")
    r.add_argument("--reasoning-effort", choices=sorted(REASONING_EFFORTS), default=None,
                   help="OpenRouter reasoning effort for this run")
    r.add_argument("--max-tokens-per-turn", "--max-tokens", dest="max_tokens_per_turn",
                   type=_positive_int, default=None,
                   help="OpenRouter max_tokens cap for each model turn")
    r.add_argument("--temperature", type=_nonnegative_float, default=None,
                   help="OpenRouter temperature for this run")
    r.add_argument("--retries", type=_nonnegative_int, default=None,
                   help="model response retries per turn")
    r.add_argument("--prior-message-turns", type=_nonnegative_int, default=None,
                   help="previous user/assistant turns to resend to sessionful model calls")
    r.add_argument("--port", type=int, default=8000)
    r.set_defaults(func=_run)

    s = sub.add_parser("score", help="print a run's leaderboard")
    s.add_argument("--run", required=True)
    s.add_argument("--allow-partial", dest="allow_partial", action="store_true",
                   help="score a run even if some games are missing/failed")
    s.set_defaults(func=_score)

    sub.add_parser("runs", help="list runs").set_defaults(func=_runs)
    sub.add_parser("agents", help="show the agent roster").set_defaults(func=_agents)

    sv = sub.add_parser("serve", help="start the observer server")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=_serve)

    w = sub.add_parser("worker", help="claim queued central runs and execute them locally")
    w.add_argument("--server", default=None, help="central site URL, e.g. https://arena.vercel.app")
    w.add_argument("--owner", default=None, help="owner name; must match submitted runs and token owner")
    w.add_argument("--token", default=None, help="bearer token from INGEST_TOKENS for this owner")
    w.add_argument("--worker-id", default=None)
    w.add_argument("--poll", type=int, default=5, help="seconds between empty-queue polls")
    w.add_argument("--lease", type=int, default=300, help="lease seconds refreshed by heartbeat")
    w.add_argument("--workers", type=int, default=8, help="local game concurrency")
    w.add_argument("--once", action="store_true", help="claim at most one job and exit")
    w.set_defaults(func=_worker)

    pu = sub.add_parser("push", help="upload a finished local run to the prod leaderboard")
    pu.add_argument("--run", default=None, help="local run id to push")
    pu.add_argument("--all", action="store_true",
                    help="push every local run with status done/partial")
    pu.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="show what would be uploaded without writing")
    pu.add_argument("--force", action="store_true",
                    help="resend all local games (server-side no-op for ones already recorded)")
    pu.add_argument("--as", dest="as_name", default=None,
                    help="display name to register/identify as (default: hostname); ignored when "
                         "$ARENA_INGEST_TOKEN or a cached identity is used")
    pu.add_argument("--server", default=None,
                    help="central site URL (default $ARENA_SERVER_URL else the prod Vercel host)")
    pu.set_defaults(func=_push)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
