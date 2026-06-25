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
            if key in agent and key not in fields:
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


def _run(args):
    from .batch import run_batch
    rid = args.run_id or f"run_{args.seed}"
    caps = _run_caps_from_args(args)
    run_batch(game=args.game, n_games=args.games, seed_base=args.seed, run_id=rid,
              workers=args.workers, discussion_rounds=args.rounds, deck_preset=args.deck,
              caps=caps)
    print(f"\nDone. View at http://localhost:{args.port}/observer.html  (run: {rid})")
    print(f"Score with: python -m arena.cli score --run {rid}")


def _score(args):
    from .score import score_run
    from . import store
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
PROD_SERVER_URL = "https://persuasion-arena.vercel.app"

PUSHABLE_RUN_STATUSES = {"done", "partial"}


def push_token_from_env(env: dict | None = None) -> str | None:
    """The bearer token push sends to /api/runs/import. Sourced from $ARENA_INGEST_TOKEN ONLY (no
    --token CLI arg, by design). Returns None when unset/blank."""
    env = os.environ if env is None else env
    token = (env.get("ARENA_INGEST_TOKEN") or "").strip()
    return token or None


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


def _push_recompute(prior_prod_count: int, local_game_players: int) -> dict:
    """Recompute prod ratings AFTER the upload, with the four disaster guardrails. Assumes
    DATABASE_URL already points at prod. prior_prod_count is the prod game_players count read
    BEFORE the upload, so the shrink guard compares against a true pre-upload baseline."""
    from . import rating, store
    current_prod = store.game_player_count()
    rating.guard_recompute(local_game_players, prior_prod_count, current_prod)
    return rating.recompute()


def _push(args):
    import json
    from . import rating, store

    # Capture any prod URL the env may carry, then POP DATABASE_URL so the READ phase
    # (store.get_run / get_game / distinct_gids) hits the LOCAL SQLite, not prod Neon.
    prod_url = os.environ.pop("DATABASE_URL", None) or os.environ.get("ARENA_PROD_DATABASE_URL")
    args.server = (args.server or os.environ.get("ARENA_SERVER_URL") or PROD_SERVER_URL).rstrip("/")
    token = push_token_from_env()
    if not token:
        print("ERROR: $ARENA_INGEST_TOKEN is not set. The import endpoint fails closed without it.")
        raise SystemExit(2)

    if args.all:
        run_ids = [r["id"] for r in store.list_runs() if r["status"] in PUSHABLE_RUN_STATUSES]
        if not run_ids:
            print("no local runs with status done/partial to push")
            return
    else:
        if not args.run:
            print("ERROR: pass --run RUN_ID (or --all to push every done/partial local run)")
            raise SystemExit(2)
        run_ids = [args.run]

    # Pre-upload prod baseline for the shrink guard (guardrail #2): read prod's game_players
    # count BEFORE uploading, so the post-upload count can be compared against a true prior.
    # Briefly point at prod, then restore local (popped) for the read/upload phase.
    prior_prod_count = 0
    if prod_url and not args.dry_run and not args.no_recompute:
        os.environ["DATABASE_URL"] = prod_url
        try:
            prior_prod_count = store.game_player_count()
        except Exception as e:
            print(f"push: WARNING could not read prod baseline count ({e}); shrink guard treats prior as 0")
        finally:
            os.environ.pop("DATABASE_URL", None)

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
        print("push: dry-run, skipping recompute")
        return
    if args.no_recompute:
        print("push: --no-recompute, skipping leaderboard rebuild")
        return

    # RECOMPUTE PHASE — point store at prod Neon, snapshot first, guard, then rebuild.
    local_game_players = store.game_player_count()  # local count (DATABASE_URL still unset here)
    if not prod_url:
        print("push: no prod DATABASE_URL (or ARENA_PROD_DATABASE_URL) available — "
              "skipping recompute. Set it (prod-scoped) to rebuild the leaderboard.")
        return
    os.environ["DATABASE_URL"] = prod_url
    os.environ.setdefault("ARENA_PG_STATEMENT_TIMEOUT_MS", "60000")
    # Pin the schema recompute writes to the one the JS leaderboard reads (default 'public'),
    # so a stray ARENA_PG_SEARCH_PATH can't make recompute populate a schema the live board
    # can't see (the "silent empty board" disaster). Operator can still override deliberately.
    os.environ.setdefault("ARENA_PG_SEARCH_PATH", "public")
    try:
        snapshot = rating.rating_snapshot()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = os.path.abspath(f"arena-backup-{stamp}-pre-recompute.json")
        with open(backup_path, "w") as fh:
            json.dump(snapshot, fh, indent=2, default=str)
        print(f"push: wrote ratings snapshot -> {backup_path}")
        try:
            result = _push_recompute(prior_prod_count, local_game_players)
        except rating.RecomputeAborted as e:
            print(f"push: RECOMPUTE ABORTED — {e}")
            print(f"push: prod ratings UNCHANGED; snapshot kept at {backup_path}")
            raise SystemExit(3)
        print(f"push: recompute done -> {result}")
    finally:
        os.environ.pop("DATABASE_URL", None)


def main():
    p = argparse.ArgumentParser(prog="arena")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="play a run of N games")
    r.add_argument("--game", default="onuw", choices=["onuw", "avalon"])
    r.add_argument("--games", type=int, default=20)
    r.add_argument("--seed", type=int, default=9000)
    r.add_argument("--run-id", dest="run_id", default=None)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--rounds", type=_positive_int, default=5, help="max discussion rounds (ends early on all-pass)")
    r.add_argument("--deck", default="arena", help="ONUW deck preset: arena, classic, or tanner")
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
                    help="show what would be uploaded without writing or recomputing")
    pu.add_argument("--force", action="store_true",
                    help="resend all local games (server-side no-op for ones already recorded)")
    pu.add_argument("--no-recompute", dest="no_recompute", action="store_true",
                    help="upload games but skip the prod leaderboard recompute")
    pu.add_argument("--server", default=None,
                    help="central site URL (default $ARENA_SERVER_URL else the prod Vercel host)")
    pu.set_defaults(func=_push)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
