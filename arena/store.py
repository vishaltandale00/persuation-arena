"""Persistence. A run is a batch of games; each game stores its full transcript JSON
(the observer's shape) plus structured rows for scoring.

Dual backend, chosen at call time by the DATABASE_URL env var:
  - unset  -> SQLite at STORE_DIR/arena.db (local; the existing behaviour, unchanged).
  - set    -> Postgres via psycopg3 (the central Vercel+Neon deployment).

The two backends share one set of SQL statements; only the parameter placeholder differs
('?' for SQLite, '%s' for Postgres). Both support UPSERT (ON CONFLICT) and RETURNING, so the
write functions are backend-agnostic. The federated/central deployment relies on two semantics
baked in here:
  - games are FIRST-WRITER-WINS (ON CONFLICT (run_id,gid) DO NOTHING): a re-published game never
    overwrites a recorded outcome. Games are LLM-played and non-deterministic, so a reclaimed
    re-run's "game 7" is a *different* game that legitimately keys to the same (run_id,7); keeping
    the first write is what makes ingest idempotent-by-key.
  - run status is MONOTONIC: a late/duplicate save_run('running') can never regress a finished run
    back to 'running', and update_run_status never un-does a 'done'.
The DATABASE_URL check is per call (never frozen at import) so tests can point at a Postgres by
setting the env var after import.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
from pathlib import Path

from .config import STORE_DIR

DB_PATH = STORE_DIR / "arena.db"

# --- schema ------------------------------------------------------------------
# SQLite: a single script (run on every conn() — cheap, local).
SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, game TEXT, label TEXT, status TEXT, n_games INTEGER,
  players INTEGER, seed_base INTEGER, created TEXT, agents_json TEXT,
  submitter TEXT, created_utc TEXT
);
CREATE TABLE IF NOT EXISTS games (
  run_id TEXT, gid INTEGER, seed INTEGER, winner_team TEXT, line TEXT,
  transcript_json TEXT, PRIMARY KEY (run_id, gid)
);
CREATE TABLE IF NOT EXISTS game_players (
  run_id TEXT, gid INTEGER, seat INTEGER, agent TEXT, model TEXT,
  dealt_role TEXT, end_role TEXT, team TEXT, won INTEGER,
  calls INTEGER DEFAULT 0, forfeits INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, run_id TEXT UNIQUE, owner TEXT, status TEXT,
  game TEXT, n_games INTEGER, seed_base INTEGER, rounds INTEGER, players INTEGER,
  agents_json TEXT, created_utc TEXT, updated_utc TEXT,
  lease_expires_utc TEXT, heartbeat_utc TEXT, worker_id TEXT, last_error TEXT
);
"""

# Postgres: created once via scripts/init_db.py or store.init_schema() (NOT per connection —
# Neon's PgBouncer transaction-mode pooler dislikes per-conn DDL). Statements run one at a time.
PG_SCHEMA_STMTS = [
    """CREATE TABLE IF NOT EXISTS runs (
         id TEXT PRIMARY KEY, game TEXT, label TEXT, status TEXT, n_games INTEGER,
         players INTEGER, seed_base BIGINT, created TEXT, agents_json TEXT,
         submitter TEXT, created_utc TEXT
       )""",
    """CREATE TABLE IF NOT EXISTS games (
         run_id TEXT, gid INTEGER, seed BIGINT, winner_team TEXT, line TEXT,
         transcript_json TEXT, PRIMARY KEY (run_id, gid)
       )""",
    """CREATE TABLE IF NOT EXISTS game_players (
         run_id TEXT, gid INTEGER, seat INTEGER, agent TEXT, model TEXT,
         dealt_role TEXT, end_role TEXT, team TEXT, won INTEGER,
         calls INTEGER DEFAULT 0, forfeits INTEGER DEFAULT 0
       )""",
    """CREATE TABLE IF NOT EXISTS jobs (
         id TEXT PRIMARY KEY, run_id TEXT UNIQUE, owner TEXT, status TEXT,
         game TEXT, n_games INTEGER, seed_base BIGINT, rounds INTEGER, players INTEGER,
         agents_json TEXT, created_utc TEXT, updated_utc TEXT,
         lease_expires_utc TEXT, heartbeat_utc TEXT, worker_id TEXT, last_error TEXT
       )""",
]

# Columns added after the original schema shipped; ALTER-added on open so old SQLite DBs upgrade.
_MIGRATIONS = {
    "game_players": [("calls", "INTEGER DEFAULT 0"), ("forfeits", "INTEGER DEFAULT 0")],
    "runs": [("submitter", "TEXT"), ("created_utc", "TEXT")],
}


def _is_pg() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _ph() -> str:
    return "%s" if _is_pg() else "?"


def _utcnow() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_after(seconds: int) -> str:
    return (_dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=seconds)).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")


def _job(row) -> dict | None:
    if not row:
        return None
    d = dict(row)
    d["agents"] = json.loads(d.pop("agents_json"))
    return d


def _migrate(c: sqlite3.Connection) -> None:
    for table, cols in _MIGRATIONS.items():
        have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, decl in cols:
            if name not in have:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def conn():
    """Open a fresh connection to the active backend. Used as `with conn() as c:` — both backends
    commit on clean exit (psycopg also closes, which is the right per-invocation serverless pattern).
    """
    if _is_pg():
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(SQLITE_SCHEMA)
    _migrate(c)
    return c


def init_schema() -> None:
    """Idempotently create the schema on the active backend. SQLite does this in conn() too;
    Postgres needs this called once (init_db.py / test setup)."""
    if _is_pg():
        with conn() as c:
            for stmt in PG_SCHEMA_STMTS:
                c.execute(stmt)
    else:
        with conn():
            pass


def save_run(meta: dict):
    """Upsert a run. MONOTONIC: never regress a finished ('done'/'partial') run back to 'running'."""
    ph = _ph()
    with conn() as c:
        c.execute(
            f"INSERT INTO runs (id,game,label,status,n_games,players,seed_base,created,agents_json,submitter,created_utc) "
            f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph}) "
            f"ON CONFLICT (id) DO UPDATE SET "
            f"  game=excluded.game, label=excluded.label, "
            f"  status=CASE WHEN runs.status IN ('done','partial') THEN runs.status ELSE excluded.status END, "
            f"  n_games=excluded.n_games, players=excluded.players, seed_base=excluded.seed_base, "
            f"  created=excluded.created, agents_json=excluded.agents_json, "
            f"  submitter=excluded.submitter, created_utc=excluded.created_utc",
            (meta["id"], meta["game"], meta["label"], meta["status"], meta["n_games"],
             meta["players"], meta["seed_base"], meta["created"], json.dumps(meta["agents"]),
             meta.get("submitter"), meta.get("created_utc")),
        )


def update_run_status(run_id: str, status: str):
    """Set a run's status. MONOTONIC: never un-do a 'done' run (so a late re-publish can't reopen it)."""
    ph = _ph()
    with conn() as c:
        c.execute(f"UPDATE runs SET status={ph} WHERE id={ph} AND status != 'done'", (status, run_id))


def save_game(run_id: str, gid: int, transcript: dict, agents: list[dict]):
    """Persist one finished game. FIRST-WRITER-WINS: if (run_id,gid) already exists, keep it and
    skip the player rows too (a reclaimed re-run's game is a *different* valid game; we don't clobber
    the recorded outcome). Game row + its player rows are written in one transaction.
    """
    ph = _ph()
    with conn() as c:
        cur = c.execute(
            f"INSERT INTO games (run_id,gid,seed,winner_team,line,transcript_json) "
            f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph}) "
            f"ON CONFLICT (run_id,gid) DO NOTHING RETURNING gid",
            (run_id, gid, transcript["seed"], transcript["winner_team"],
             transcript["outcome"]["text"], json.dumps(transcript)),
        )
        inserted = cur.fetchone() is not None
        if not inserted:
            return False  # game already recorded; first write wins
        c.execute(f"DELETE FROM game_players WHERE run_id={ph} AND gid={ph}", (run_id, gid))
        for p in transcript["players"]:
            agent = agents[p["seat"]]
            c.execute(
                f"INSERT INTO game_players (run_id,gid,seat,agent,model,dealt_role,end_role,team,won,calls,forfeits) "
                f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})",
                (run_id, gid, p["seat"], agent["name"], agent["model"],
                 p["dealt"], p["end"], p["team"], 1 if p["won"] else 0,
                 int(p.get("calls", 0)), int(p.get("forfeits", 0))),
            )
        return True


def enqueue_job(job: dict) -> dict:
    """Create a central queued job and its visible run row."""
    now = job.get("created_utc") or _utcnow()
    agents = job["agents"]
    save_run({
        "id": job["run_id"], "game": job["game"], "label": job["label"], "status": "queued",
        "n_games": job["n_games"], "players": job["players"], "seed_base": job["seed_base"],
        "created": job.get("created") or now[:16].replace("T", " "), "created_utc": now,
        "submitter": job["owner"], "agents": agents,
    })
    ph = _ph()
    with conn() as c:
        c.execute(
            f"INSERT INTO jobs (id,run_id,owner,status,game,n_games,seed_base,rounds,players,"
            f"agents_json,created_utc,updated_utc,lease_expires_utc,heartbeat_utc,worker_id,last_error) "
            f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph}) "
            f"ON CONFLICT (run_id) DO UPDATE SET "
            f"  owner=excluded.owner, status=excluded.status, game=excluded.game, "
            f"  n_games=excluded.n_games, seed_base=excluded.seed_base, rounds=excluded.rounds, "
            f"  players=excluded.players, agents_json=excluded.agents_json, updated_utc=excluded.updated_utc, "
            f"  lease_expires_utc=NULL, heartbeat_utc=NULL, worker_id=NULL, last_error=NULL",
            (job["id"], job["run_id"], job["owner"], "queued", job["game"], job["n_games"],
             job["seed_base"], job["rounds"], job["players"], json.dumps(agents), now, now,
             None, None, None, None),
        )
    return get_job(job["id"])


def get_job(job_id: str) -> dict | None:
    ph = _ph()
    with conn() as c:
        row = c.execute(f"SELECT * FROM jobs WHERE id={ph}", (job_id,)).fetchone()
        return _job(row)


def claim_job(owner: str, worker_id: str, lease_seconds: int = 300) -> dict | None:
    """Atomically claim the next queued or expired job for this owner."""
    now, lease_until, ph = _utcnow(), _utc_after(lease_seconds), _ph()
    with conn() as c:
        if _is_pg():
            row = c.execute(
                """
                WITH next_job AS (
                  SELECT id FROM jobs
                  WHERE owner=%s
                    AND (status='queued' OR (status='running' AND lease_expires_utc < %s))
                  ORDER BY created_utc
                  FOR UPDATE SKIP LOCKED
                  LIMIT 1
                )
                UPDATE jobs
                SET status='running', worker_id=%s, lease_expires_utc=%s,
                    heartbeat_utc=%s, updated_utc=%s
                WHERE id=(SELECT id FROM next_job)
                RETURNING *
                """,
                (owner, now, worker_id, lease_until, now, now),
            ).fetchone()
        else:
            c.execute("BEGIN IMMEDIATE")
            found = c.execute(
                f"SELECT id FROM jobs WHERE owner={ph} "
                f"AND (status='queued' OR (status='running' AND lease_expires_utc < {ph})) "
                f"ORDER BY created_utc LIMIT 1",
                (owner, now),
            ).fetchone()
            if not found:
                return None
            c.execute(
                f"UPDATE jobs SET status='running', worker_id={ph}, lease_expires_utc={ph}, "
                f"heartbeat_utc={ph}, updated_utc={ph} WHERE id={ph}",
                (worker_id, lease_until, now, now, found["id"]),
            )
            row = c.execute(f"SELECT * FROM jobs WHERE id={ph}", (found["id"],)).fetchone()
        if row:
            c.execute(f"UPDATE runs SET status={ph} WHERE id={ph} AND status != 'done'",
                      ("running", row["run_id"]))
        return _job(row)


def heartbeat_job(job_id: str, worker_id: str, lease_seconds: int = 300) -> bool:
    now, lease_until, ph = _utcnow(), _utc_after(lease_seconds), _ph()
    with conn() as c:
        cur = c.execute(
            f"UPDATE jobs SET lease_expires_utc={ph}, heartbeat_utc={ph}, updated_utc={ph} "
            f"WHERE id={ph} AND worker_id={ph} AND status='running'",
            (lease_until, now, now, job_id, worker_id),
        )
        return cur.rowcount > 0


def finish_job(job_id: str, worker_id: str, status: str, error: str | None = None) -> bool:
    if status not in {"done", "partial", "failed"}:
        raise ValueError(f"invalid job status: {status}")
    now, ph = _utcnow(), _ph()
    run_status = "done" if status == "done" else "partial"
    with conn() as c:
        job = c.execute(f"SELECT run_id FROM jobs WHERE id={ph} AND worker_id={ph}",
                        (job_id, worker_id)).fetchone()
        if not job:
            return False
        cur = c.execute(
            f"UPDATE jobs SET status={ph}, updated_utc={ph}, lease_expires_utc=NULL, "
            f"heartbeat_utc={ph}, last_error={ph} WHERE id={ph} AND worker_id={ph}",
            (status, now, now, error, job_id, worker_id),
        )
        c.execute(f"UPDATE runs SET status={ph} WHERE id={ph} AND status != 'done'",
                  (run_status, job["run_id"]))
        return cur.rowcount > 0


def list_runs() -> list[dict]:
    ph = _ph()
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM runs ORDER BY created_utc DESC NULLS LAST, created DESC").fetchall()
        out = []
        for r in rows:
            agents = json.loads(r["agents_json"])
            wins = c.execute(
                f"SELECT team, COUNT(*) n FROM ("
                f"  SELECT DISTINCT run_id,gid,winner_team team FROM games WHERE run_id={ph}"
                f") sub GROUP BY team", (r["id"],)).fetchall()
            split = {"good": 0, "evil": 0}
            for w in wins:
                split[w["team"]] = w["n"]
            out.append({**dict(r), "agents": agents, "team_split": split})
        return out


def get_run(run_id: str) -> dict | None:
    ph = _ph()
    with conn() as c:
        r = c.execute(f"SELECT * FROM runs WHERE id={ph}", (run_id,)).fetchone()
        if not r:
            return None
        agents = json.loads(r["agents_json"])
        games = c.execute(
            f"SELECT gid,seed,winner_team,line FROM games WHERE run_id={ph} ORDER BY gid", (run_id,)).fetchall()
        # per-agent wins across the run
        wins = {a["name"]: 0 for a in agents}
        wr = c.execute(
            f"SELECT agent, SUM(won) w FROM game_players WHERE run_id={ph} GROUP BY agent", (run_id,)).fetchall()
        for row in wr:
            wins[row["agent"]] = row["w"] or 0
        split = {"good": 0, "evil": 0}
        for g in games:
            split[g["winner_team"]] = split.get(g["winner_team"], 0) + 1
        return {**dict(r), "agents": agents, "wins": wins, "team_split": split,
                "games": [dict(g) for g in games]}


def get_game(run_id: str, gid: int) -> dict | None:
    ph = _ph()
    with conn() as c:
        r = c.execute(
            f"SELECT transcript_json FROM games WHERE run_id={ph} AND gid={ph}", (run_id, gid)).fetchone()
        return json.loads(r["transcript_json"]) if r else None


def player_rows(run_id: str) -> list[dict]:
    ph = _ph()
    with conn() as c:
        rows = c.execute(
            f"SELECT agent, team, won, dealt_role, end_role, calls, forfeits "
            f"FROM game_players WHERE run_id={ph}", (run_id,)).fetchall()
        return [dict(r) for r in rows]


def distinct_gids(run_id: str) -> list[int]:
    """Game ids actually persisted for a run (used to detect incomplete/partial runs and, in the
    federated worker, to resume by skipping games already published)."""
    ph = _ph()
    with conn() as c:
        rows = c.execute(
            f"SELECT DISTINCT gid FROM games WHERE run_id={ph} ORDER BY gid", (run_id,)).fetchall()
        return [r["gid"] for r in rows]
