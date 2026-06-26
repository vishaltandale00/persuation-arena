"""Small admin tool for the Arena store (whichever backend DATABASE_URL selects).

  dump   --out FILE                 JSON snapshot of every table (back up before destructive ops)
  list                              one line per run (id, game, status, k, games saved, agents)
  delete --ids a,b,c   [--yes]      delete those runs + all their child rows
  delete --all         [--yes]      delete ALL runs + child rows (clean slate)

Without --yes it prints what it WOULD delete and exits (dry run). Targets prod Neon when
DATABASE_URL points there, so it always prints the backend host first.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from arena import store

# child tables keyed by run_id, deleted before the runs row; turn_replies is keyed by turn_id.
RUN_TABLES = ["run_signups", "run_events", "game_players", "games", "turns"]
ALL_TABLES = ["runs", "games", "game_players", "run_events",
              "run_signups", "turns", "turn_replies", "agents"]


def _backend() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        return f"sqlite ({store.DB_PATH})"
    host = url.split("@")[-1].split("/")[0] if "@" in url else "postgres"
    return f"postgres ({host})"


def _ids_in_db() -> list[str]:
    return [r["id"] for r in store.list_runs()]


def cmd_dump(args) -> int:
    snap = {}
    with store.conn() as c:
        for t in ALL_TABLES:
            try:
                rows = c.execute(f"SELECT * FROM {t}").fetchall()
                snap[t] = [dict(r) for r in rows]
            except Exception as e:
                snap[t] = {"_error": f"{type(e).__name__}: {e}"}
    Path(args.out).write_text(json.dumps(snap, default=str, indent=2))
    counts = {t: (len(v) if isinstance(v, list) else "err") for t, v in snap.items()}
    print(f"backend: {_backend()}")
    print(f"wrote {args.out}")
    print("rows:", json.dumps(counts))
    return 0


def cmd_list(args) -> int:
    print(f"backend: {_backend()}")
    for r in store.list_runs():
        names = [a.get("name") for a in (r.get("agents") or [])]
        n = len(store.distinct_gids(r["id"]))
        print(f"- {r['id']:30} {r['game']:6} {r['status']:9} k={r['n_games']:<3} "
              f"games={n:<3} {names}")
    return 0


def cmd_delete(args) -> int:
    print(f"backend: {_backend()}")
    targets = _ids_in_db() if args.all else [x.strip() for x in (args.ids or "").split(",") if x.strip()]
    if not targets:
        print("no target run ids (use --ids a,b or --all)")
        return 1
    print(f"target runs ({len(targets)}): {targets}")
    if not args.yes:
        print("\nDRY RUN — re-run with --yes to actually delete.")
        return 0

    ph = store._ph()
    placeholders = ",".join([ph] * len(targets))
    deleted = {}
    with store.conn() as c:
        # turn_replies is keyed by turn_id, not run_id
        cur = c.execute(
            f"DELETE FROM turn_replies WHERE turn_id IN "
            f"(SELECT id FROM turns WHERE run_id IN ({placeholders}))", tuple(targets))
        deleted["turn_replies"] = cur.rowcount
        for t in RUN_TABLES:
            cur = c.execute(f"DELETE FROM {t} WHERE run_id IN ({placeholders})", tuple(targets))
            deleted[t] = cur.rowcount
        cur = c.execute(f"DELETE FROM runs WHERE id IN ({placeholders})", tuple(targets))
        deleted["runs"] = cur.rowcount
    print("deleted rows:", json.dumps(deleted))
    print(f"runs remaining: {len(_ids_in_db())}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="arena_admin")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump"); d.add_argument("--out", required=True); d.set_defaults(func=cmd_dump)
    sub.add_parser("list").set_defaults(func=cmd_list)
    x = sub.add_parser("delete")
    x.add_argument("--ids"); x.add_argument("--all", action="store_true")
    x.add_argument("--yes", action="store_true")
    x.set_defaults(func=cmd_delete)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
