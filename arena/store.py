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
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import STORE_DIR

DB_PATH = STORE_DIR / "arena.db"

# --- schema ------------------------------------------------------------------
# SQLite: a single script (run on every conn() — cheap, local).
SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, game TEXT, label TEXT, status TEXT, n_games INTEGER,
  players INTEGER, seed_base INTEGER, created TEXT, agents_json TEXT,
  submitter TEXT, created_utc TEXT, deck_preset TEXT
);
CREATE TABLE IF NOT EXISTS games (
  run_id TEXT, gid INTEGER, seed INTEGER, winner_team TEXT, line TEXT,
  transcript_json TEXT, PRIMARY KEY (run_id, gid)
);
CREATE TABLE IF NOT EXISTS game_players (
  run_id TEXT, gid INTEGER, seat INTEGER, agent TEXT, model TEXT,
  dealt_role TEXT, end_role TEXT, team TEXT, won INTEGER,
  calls INTEGER DEFAULT 0, forfeits INTEGER DEFAULT 0,
  agent_id TEXT, signup_id TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, run_id TEXT UNIQUE, owner TEXT, status TEXT,
  game TEXT, n_games INTEGER, seed_base INTEGER, rounds INTEGER, players INTEGER,
  agents_json TEXT, created_utc TEXT, updated_utc TEXT,
  lease_expires_utc TEXT, heartbeat_utc TEXT, worker_id TEXT, last_error TEXT,
  deck_preset TEXT
);
CREATE TABLE IF NOT EXISTS agents (
  id TEXT PRIMARY KEY, display_name TEXT, token_hash TEXT UNIQUE,
  protocol_version TEXT, sdk_version TEXT, created_utc TEXT, last_seen_utc TEXT, status TEXT,
  declared_model TEXT, declared_harness TEXT
);
CREATE TABLE IF NOT EXISTS run_signups (
  id TEXT PRIMARY KEY, run_id TEXT, agent_id TEXT, status TEXT, seat INTEGER,
  created_utc TEXT, updated_utc TEXT, waiting_expires_utc TEXT, ready_deadline_utc TEXT,
  last_poll_utc TEXT, last_event_id TEXT, max_concurrent_turns INTEGER DEFAULT 1,
  UNIQUE (run_id, agent_id)
);
CREATE TABLE IF NOT EXISTS run_events (
  id TEXT PRIMARY KEY, run_id TEXT, game_instance_id TEXT, seq INTEGER,
  visibility TEXT, target_signup_id TEXT, phase TEXT, type TEXT, payload_json TEXT, created_utc TEXT
);
CREATE TABLE IF NOT EXISTS turns (
  id TEXT PRIMARY KEY, run_id TEXT, signup_id TEXT, game_instance_id TEXT, seat INTEGER,
  phase TEXT, action_kind TEXT, observation_json TEXT, legal_action_json TEXT,
  status TEXT, deadline_utc TEXT, created_utc TEXT, claimed_utc TEXT
);
CREATE TABLE IF NOT EXISTS turn_replies (
  turn_id TEXT PRIMARY KEY, signup_id TEXT, action_json TEXT, reasoning TEXT,
  client_ms INTEGER, accepted INTEGER, created_utc TEXT
);
CREATE TABLE IF NOT EXISTS ratings (
  identity_key TEXT PRIMARY KEY, agent_id TEXT, display_name TEXT,
  declared_model TEXT, declared_harness TEXT, skill REAL, rd REAL, elo REAL,
  games INTEGER, wins INTEGER, forfeit_rate REAL, provisional INTEGER, updated_utc TEXT
);
CREATE TABLE IF NOT EXISTS rating_events (
  id TEXT PRIMARY KEY, identity_key TEXT, run_id TEXT, gid INTEGER, seat INTEGER,
  dealt_role TEXT, objective_group TEXT, won INTEGER, pre_skill REAL, post_skill REAL,
  delta REAL, expected REAL, d_r REAL, resistance REAL, k REAL, damped REAL, created_utc TEXT
);
CREATE TABLE IF NOT EXISTS role_difficulty (
  bucket TEXT, role TEXT, w INTEGER, n INTEGER, base_rate REAL, d_r REAL, updated_utc TEXT,
  PRIMARY KEY (bucket, role)
);
"""

# Postgres: created once via scripts/init_db.py or store.init_schema() (NOT per connection —
# Neon's PgBouncer transaction-mode pooler dislikes per-conn DDL). Statements run one at a time.
PG_SCHEMA_STMTS = [
    """CREATE TABLE IF NOT EXISTS runs (
         id TEXT PRIMARY KEY, game TEXT, label TEXT, status TEXT, n_games INTEGER,
         players INTEGER, seed_base BIGINT, created TEXT, agents_json TEXT,
         submitter TEXT, created_utc TEXT, deck_preset TEXT, coordinator_url TEXT
       )""",
    """CREATE TABLE IF NOT EXISTS games (
         run_id TEXT, gid INTEGER, seed BIGINT, winner_team TEXT, line TEXT,
         transcript_json TEXT, PRIMARY KEY (run_id, gid)
       )""",
    """CREATE TABLE IF NOT EXISTS game_players (
         run_id TEXT, gid INTEGER, seat INTEGER, agent TEXT, model TEXT,
         dealt_role TEXT, end_role TEXT, team TEXT, won INTEGER,
         calls INTEGER DEFAULT 0, forfeits INTEGER DEFAULT 0,
         agent_id TEXT, signup_id TEXT
       )""",
    """CREATE TABLE IF NOT EXISTS jobs (
         id TEXT PRIMARY KEY, run_id TEXT UNIQUE, owner TEXT, status TEXT,
         game TEXT, n_games INTEGER, seed_base BIGINT, rounds INTEGER, players INTEGER,
         agents_json TEXT, created_utc TEXT, updated_utc TEXT,
         lease_expires_utc TEXT, heartbeat_utc TEXT, worker_id TEXT, last_error TEXT,
         deck_preset TEXT
       )""",
    """CREATE TABLE IF NOT EXISTS agents (
         id TEXT PRIMARY KEY, display_name TEXT, token_hash TEXT UNIQUE,
         protocol_version TEXT, sdk_version TEXT, created_utc TEXT, last_seen_utc TEXT, status TEXT,
         declared_model TEXT, declared_harness TEXT
       )""",
    """CREATE TABLE IF NOT EXISTS run_signups (
         id TEXT PRIMARY KEY, run_id TEXT, agent_id TEXT, status TEXT, seat INTEGER,
         created_utc TEXT, updated_utc TEXT, waiting_expires_utc TEXT, ready_deadline_utc TEXT,
         last_poll_utc TEXT, last_event_id TEXT, max_concurrent_turns INTEGER DEFAULT 1,
         UNIQUE (run_id, agent_id)
       )""",
    """CREATE TABLE IF NOT EXISTS run_events (
         id TEXT PRIMARY KEY, run_id TEXT, game_instance_id TEXT, seq INTEGER,
         visibility TEXT, target_signup_id TEXT, phase TEXT, type TEXT, payload_json TEXT, created_utc TEXT
       )""",
    """CREATE TABLE IF NOT EXISTS turns (
         id TEXT PRIMARY KEY, run_id TEXT, signup_id TEXT, game_instance_id TEXT, seat INTEGER,
         phase TEXT, action_kind TEXT, observation_json TEXT, legal_action_json TEXT,
         status TEXT, deadline_utc TEXT, created_utc TEXT, claimed_utc TEXT
       )""",
    """CREATE TABLE IF NOT EXISTS turn_replies (
         turn_id TEXT PRIMARY KEY, signup_id TEXT, action_json TEXT, reasoning TEXT,
         client_ms INTEGER, accepted INTEGER, created_utc TEXT
       )""",
    """CREATE TABLE IF NOT EXISTS ratings (
         identity_key TEXT PRIMARY KEY, agent_id TEXT, display_name TEXT,
         declared_model TEXT, declared_harness TEXT, skill DOUBLE PRECISION, rd DOUBLE PRECISION,
         elo DOUBLE PRECISION, games INTEGER, wins INTEGER, forfeit_rate DOUBLE PRECISION,
         provisional INTEGER, updated_utc TEXT
       )""",
    """CREATE TABLE IF NOT EXISTS rating_events (
         id TEXT PRIMARY KEY, identity_key TEXT, run_id TEXT, gid INTEGER, seat INTEGER,
         dealt_role TEXT, objective_group TEXT, won INTEGER, pre_skill DOUBLE PRECISION,
         post_skill DOUBLE PRECISION, delta DOUBLE PRECISION, expected DOUBLE PRECISION,
         d_r DOUBLE PRECISION, resistance DOUBLE PRECISION, k DOUBLE PRECISION,
         damped DOUBLE PRECISION, created_utc TEXT
       )""",
    """CREATE TABLE IF NOT EXISTS role_difficulty (
         bucket TEXT, role TEXT, w INTEGER, n INTEGER, base_rate DOUBLE PRECISION,
         d_r DOUBLE PRECISION, updated_utc TEXT, PRIMARY KEY (bucket, role)
       )""",
]

# Columns added after the original schema shipped; ALTER-added on open so old SQLite DBs upgrade.
_MIGRATIONS = {
    "game_players": [("calls", "INTEGER DEFAULT 0"), ("forfeits", "INTEGER DEFAULT 0"),
                     ("agent_id", "TEXT"), ("signup_id", "TEXT")],
    "runs": [("submitter", "TEXT"), ("created_utc", "TEXT"), ("deck_preset", "TEXT"),
             ("coordinator_url", "TEXT")],
    "jobs": [("deck_preset", "TEXT")],
    "agents": [("declared_model", "TEXT"), ("declared_harness", "TEXT")],
}

PG_MIGRATION_STMTS = [
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS deck_preset TEXT",
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS coordinator_url TEXT",
    "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS deck_preset TEXT",
    "ALTER TABLE game_players ADD COLUMN IF NOT EXISTS agent_id TEXT",
    "ALTER TABLE game_players ADD COLUMN IF NOT EXISTS signup_id TEXT",
    "ALTER TABLE agents ADD COLUMN IF NOT EXISTS declared_model TEXT",
    "ALTER TABLE agents ADD COLUMN IF NOT EXISTS declared_harness TEXT",
]


def _is_pg() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _ph() -> str:
    return "%s" if _is_pg() else "?"


def _utcnow() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _utc_after(seconds: int) -> str:
    return (_dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=seconds)).isoformat(
        timespec="microseconds").replace("+00:00", "Z")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _apply_pg_session_settings(c, statement_timeout: int) -> None:
    search_path = os.environ.get("ARENA_PG_SEARCH_PATH", "").strip()
    if search_path:
        names = [name.strip() for name in search_path.split(",") if name.strip()]
        if not names or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in names):
            raise ValueError("ARENA_PG_SEARCH_PATH must contain comma-separated identifiers")
        quoted = ", ".join(f'"{name}"' for name in names)
        c.execute(f"SET search_path TO {quoted}")
    if statement_timeout > 0:
        c.execute(f"SET statement_timeout = {int(statement_timeout)}")


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


def _backfill_connected_game_player_identities(c) -> dict:
    """Repair legacy connected game rows by matching the saved player name to a unique run roster row."""
    ph = _ph()
    runs = c.execute(
        "SELECT DISTINCT r.id, r.agents_json "
        "FROM runs r JOIN game_players gp ON gp.run_id=r.id "
        "WHERE gp.model='connected-agent' AND (gp.agent_id IS NULL OR gp.signup_id IS NULL)"
    ).fetchall()
    rows_updated = 0
    ambiguous_names = 0
    usable_names = 0
    for run in runs:
        try:
            agents = json.loads(run["agents_json"] or "[]")
        except (TypeError, ValueError):
            agents = []
        counts: dict[str, int] = {}
        for agent in agents:
            name = agent.get("name")
            if name:
                counts[name] = counts.get(name, 0) + 1
        ambiguous_names += sum(1 for n in counts.values() if n > 1)
        for agent in agents:
            name = agent.get("name")
            if not name or counts.get(name) != 1:
                continue
            agent_id = agent.get("agent_id")
            signup_id = agent.get("signup_id")
            if not (agent_id or signup_id):
                continue
            usable_names += 1
            cur = c.execute(
                f"UPDATE game_players "
                f"SET agent_id=COALESCE(agent_id,{ph}), signup_id=COALESCE(signup_id,{ph}) "
                f"WHERE run_id={ph} AND model='connected-agent' AND agent={ph} "
                f"  AND (agent_id IS NULL OR signup_id IS NULL)",
                (agent_id, signup_id, run["id"], name),
            )
            rows_updated += max(cur.rowcount or 0, 0)
    return {
        "runs_examined": len(runs),
        "usable_roster_names": usable_names,
        "ambiguous_roster_names": ambiguous_names,
        "rows_updated": rows_updated,
    }


@contextmanager
def conn():
    """Open a fresh connection to the active backend. Used as `with conn() as c:` — both backends
    commit on clean exit (psycopg also closes, which is the right per-invocation serverless pattern).
    """
    if _is_pg():
        import psycopg
        from psycopg.rows import dict_row
        connect_timeout = _env_int("ARENA_PG_CONNECT_TIMEOUT_SECONDS", 8)
        statement_timeout = _env_int("ARENA_PG_STATEMENT_TIMEOUT_MS", 15000)
        c = psycopg.connect(
            os.environ["DATABASE_URL"],
            row_factory=dict_row,
            connect_timeout=connect_timeout,
        )
        _apply_pg_session_settings(c, statement_timeout)
    else:
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(DB_PATH)
        c.row_factory = sqlite3.Row
        c.executescript(SQLITE_SCHEMA)
        _migrate(c)
    try:
        with c:
            yield c
    finally:
        c.close()


def init_schema() -> None:
    """Idempotently create the schema on the active backend. SQLite does this in conn() too;
    Postgres needs this called once (init_db.py / test setup)."""
    if _is_pg():
        with conn() as c:
            for stmt in PG_SCHEMA_STMTS:
                c.execute(stmt)
            for stmt in PG_MIGRATION_STMTS:
                c.execute(stmt)
            _backfill_connected_game_player_identities(c)
    else:
        with conn() as c:
            _backfill_connected_game_player_identities(c)


def save_run(meta: dict):
    """Upsert a run. MONOTONIC: never regress a finished ('done'/'partial') run back to 'running'."""
    ph = _ph()
    with conn() as c:
        c.execute(
            f"INSERT INTO runs (id,game,label,status,n_games,players,seed_base,created,agents_json,submitter,created_utc,deck_preset) "
            f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph}) "
            f"ON CONFLICT (id) DO UPDATE SET "
            f"  game=excluded.game, label=excluded.label, "
            f"  status=CASE WHEN runs.status IN ('done','partial') THEN runs.status ELSE excluded.status END, "
            f"  n_games=excluded.n_games, players=excluded.players, seed_base=excluded.seed_base, "
            f"  created=excluded.created, agents_json=excluded.agents_json, "
            f"  submitter=excluded.submitter, created_utc=excluded.created_utc, "
            f"  deck_preset=excluded.deck_preset",
            (meta["id"], meta["game"], meta["label"], meta["status"], meta["n_games"],
             meta["players"], meta["seed_base"], meta["created"], json.dumps(meta["agents"]),
             meta.get("submitter"), meta.get("created_utc"), meta.get("deck_preset")),
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
                f"INSERT INTO game_players (run_id,gid,seat,agent,model,dealt_role,end_role,team,won,calls,forfeits,agent_id,signup_id) "
                f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})",
                (run_id, gid, p["seat"], agent["name"], agent["model"],
                 p["dealt"], p["end"], p["team"], 1 if p["won"] else 0,
                 int(p.get("calls", 0)), int(p.get("forfeits", 0)),
                 agent.get("agent_id"), agent.get("signup_id")),
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
        "submitter": job["owner"], "agents": agents, "deck_preset": job.get("deck_preset"),
    })
    ph = _ph()
    with conn() as c:
        c.execute(
            f"INSERT INTO jobs (id,run_id,owner,status,game,n_games,seed_base,rounds,players,"
            f"agents_json,created_utc,updated_utc,lease_expires_utc,heartbeat_utc,worker_id,last_error,deck_preset) "
            f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph}) "
            f"ON CONFLICT (run_id) DO UPDATE SET "
            f"  owner=excluded.owner, status=excluded.status, game=excluded.game, "
            f"  n_games=excluded.n_games, seed_base=excluded.seed_base, rounds=excluded.rounds, "
            f"  players=excluded.players, agents_json=excluded.agents_json, updated_utc=excluded.updated_utc, "
            f"  deck_preset=excluded.deck_preset, "
            f"  lease_expires_utc=NULL, heartbeat_utc=NULL, worker_id=NULL, last_error=NULL",
            (job["id"], job["run_id"], job["owner"], "queued", job["game"], job["n_games"],
             job["seed_base"], job["rounds"], job["players"], json.dumps(agents), now, now,
             None, None, None, None, job.get("deck_preset")),
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
            f"SELECT gid, seat, agent, agent_id, signup_id, model, team, won, dealt_role, end_role, calls, forfeits "
            f"FROM game_players WHERE run_id={ph}", (run_id,)).fetchall()
        return [dict(r) for r in rows]


# --- rating engine I/O (consumed by arena/rating.py) ------------------------------------------

def all_game_player_rows() -> list[dict]:
    """Every game_players row across all runs — the raw material for the rating replay."""
    with conn() as c:
        rows = c.execute(
            "SELECT run_id, gid, seat, agent, agent_id, signup_id, model, dealt_role, end_role, "
            "team, won, calls, forfeits FROM game_players").fetchall()
        return [dict(r) for r in rows]


def run_meta_map() -> dict[str, dict]:
    """run_id -> {game, players, deck_preset, created_utc, agents:[...]} for replay bucketing/ordering."""
    out: dict[str, dict] = {}
    with conn() as c:
        for r in c.execute(
                "SELECT id, game, players, deck_preset, created_utc, agents_json FROM runs").fetchall():
            d = dict(r)
            try:
                d["agents"] = json.loads(d.pop("agents_json") or "[]")
            except (TypeError, ValueError):
                d["agents"] = []
            out[d.pop("id")] = d
    return out


def backfill_connected_game_player_identities() -> dict:
    """Fill missing agent_id/signup_id on legacy connected-agent game rows."""
    with conn() as c:
        return _backfill_connected_game_player_identities(c)


def replace_ratings(difficulty: list[dict], events: list[dict], ratings: list[dict]) -> None:
    """Atomically swap in a freshly computed rating snapshot. Ratings are derived, so a full
    rebuild clears and re-writes all three tables in one transaction."""
    ph = _ph()
    with conn() as c:
        c.execute("DELETE FROM role_difficulty")
        c.execute("DELETE FROM rating_events")
        c.execute("DELETE FROM ratings")
        for d in difficulty:
            c.execute(
                f"INSERT INTO role_difficulty (bucket,role,w,n,base_rate,d_r,updated_utc) "
                f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph})",
                (d["bucket"], d["role"], d["w"], d["n"], d["base_rate"], d["d_r"], d["updated_utc"]))
        for e in events:
            c.execute(
                f"INSERT INTO rating_events (id,identity_key,run_id,gid,seat,dealt_role,objective_group,"
                f"won,pre_skill,post_skill,delta,expected,d_r,resistance,k,damped,created_utc) "
                f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})",
                (e["id"], e["identity_key"], e["run_id"], e["gid"], e["seat"], e["dealt_role"],
                 e["objective_group"], e["won"], e["pre_skill"], e["post_skill"], e["delta"],
                 e["expected"], e["d_r"], e["resistance"], e["k"], e["damped"], e["created_utc"]))
        for r in ratings:
            c.execute(
                f"INSERT INTO ratings (identity_key,agent_id,display_name,declared_model,declared_harness,"
                f"skill,rd,elo,games,wins,forfeit_rate,provisional,updated_utc) "
                f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})",
                (r["identity_key"], r["agent_id"], r["display_name"], r["declared_model"],
                 r["declared_harness"], r["skill"], r["rd"], r["elo"], r["games"], r["wins"],
                 r["forfeit_rate"], r["provisional"], r["updated_utc"]))


def leaderboard_rows() -> list[dict]:
    """The ratings snapshot, ordered by the conservative lower bound (elo - 2*173*rd)."""
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM ratings ORDER BY (elo - 2*173.0*rd) DESC, games DESC").fetchall()
        return [dict(r) for r in rows]


def rating_events_for(identity_key: str) -> list[dict]:
    """One competitor's full rating ledger, oldest first (rating history)."""
    ph = _ph()
    with conn() as c:
        rows = c.execute(
            f"SELECT * FROM rating_events WHERE identity_key={ph} ORDER BY created_utc, run_id, gid, seat",
            (identity_key,)).fetchall()
        return [dict(r) for r in rows]


def role_difficulty_map() -> dict[tuple[str, str], dict]:
    """(bucket, role) -> {w,n,base_rate,d_r} from the last recompute."""
    with conn() as c:
        return {(r["bucket"], r["role"]): dict(r)
                for r in c.execute("SELECT * FROM role_difficulty").fetchall()}


def all_rating_events() -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM rating_events").fetchall()]


def get_rating(identity_key: str) -> dict | None:
    ph = _ph()
    with conn() as c:
        return _rowdict(c.execute(
            f"SELECT * FROM ratings WHERE identity_key={ph}", (identity_key,)).fetchone())


def distinct_gids(run_id: str) -> list[int]:
    """Game ids actually persisted for a run (used to detect incomplete/partial runs and, in the
    federated worker, to resume by skipping games already published)."""
    ph = _ph()
    with conn() as c:
        rows = c.execute(
            f"SELECT DISTINCT gid FROM games WHERE run_id={ph} ORDER BY gid", (run_id,)).fetchall()
        return [r["gid"] for r in rows]


# --- connected-agent run protocol -------------------------------------------
OPEN_RUN_STATUSES = {"open", "waiting", "ready_required"}
ACTIVE_SIGNUP_STATUSES = {"waiting", "ready_required", "ready", "active"}
TERMINAL_SIGNUP_STATUSES = {"completed", "rejected", "expired", "cancelled"}


def _rowdict(row) -> dict | None:
    return dict(row) if row else None


def _json_row(row: dict | None, *cols: str) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    for col in cols:
        if out.get(col):
            out[col[:-5] if col.endswith("_json") else col] = json.loads(out[col])
    return out


def _schema_type_ok(value, expected) -> bool:
    kinds = expected if isinstance(expected, list) else [expected]
    for kind in kinds:
        if kind == "null" and value is None:
            return True
        if kind == "boolean" and isinstance(value, bool):
            return True
        if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if kind == "string" and isinstance(value, str):
            return True
        if kind == "array" and isinstance(value, list):
            return True
        if kind == "object" and isinstance(value, dict):
            return True
    return False


def _validate_json_schema(value, schema: dict | None) -> tuple[bool, str | None]:
    """Small JSON-schema subset for agent actions.

    The action schemas we serve are intentionally simple. Keeping this local avoids adding a runtime
    dependency just to reject malformed remote replies before first-writer-wins stores them.
    """
    if not schema:
        return True, None
    if "oneOf" in schema:
        errors = []
        for option in schema["oneOf"]:
            ok, err = _validate_json_schema(value, option)
            if ok:
                return True, None
            errors.append(err or "invalid")
        return False, "; ".join(errors[:2])
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            ok, _ = _validate_json_schema(value, option)
            if ok:
                return True, None
        return False, "does not match any allowed shape"
    if "enum" in schema and value not in schema["enum"]:
        return False, "not in enum"
    if "type" in schema and not _schema_type_ok(value, schema["type"]):
        return False, f"expected {schema['type']}"

    if isinstance(value, dict):
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                return False, f"missing required field {key}"
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                return False, f"unexpected field {sorted(extra)[0]}"
        for key, subschema in properties.items():
            if key in value:
                ok, err = _validate_json_schema(value[key], subschema)
                if not ok:
                    return False, f"{key}: {err}"

    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            return False, "too few items"
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            return False, "too many items"
        if schema.get("uniqueItems") and len(value) != len({json.dumps(v, sort_keys=True) for v in value}):
            return False, "items must be unique"
        if "items" in schema:
            for item in value:
                ok, err = _validate_json_schema(item, schema["items"])
                if not ok:
                    return False, f"item: {err}"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False, "below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return False, "above maximum"
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            return False, "string too short"
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            return False, "string too long"
    return True, None


def _validate_action(action: dict, legal_action: dict | None) -> tuple[bool, str | None]:
    legal_action = legal_action or {}
    ok, err = _validate_json_schema(action, legal_action.get("schema"))
    if not ok:
        return False, err
    for fields in (legal_action.get("rules") or {}).get("distinct", []):
        values = [action.get(f) for f in fields if isinstance(action, dict)]
        non_null = [v for v in values if v is not None]
        if len(non_null) != len(set(non_null)):
            return False, f"{','.join(fields)} must be distinct"
    return True, None


def create_connected_run(meta: dict) -> dict:
    """Create a concrete run that connected agents can sign up for."""
    now = meta.get("created_utc") or _utcnow()
    run_id = meta["id"]
    save_run({
        "id": run_id,
        "game": meta["game"],
        "label": meta["label"],
        "status": meta.get("status", "open"),
        "n_games": meta["n_games"],
        "players": meta["players"],
        "seed_base": meta["seed_base"],
        "created": meta.get("created") or now[:16].replace("T", " "),
        "created_utc": now,
        "submitter": meta.get("submitter", "connected"),
        "agents": meta.get("agents", []),
        "deck_preset": meta.get("deck_preset"),
    })
    return get_run(run_id)


def list_open_runs(game: str | None = None) -> list[dict]:
    ph = _ph()
    with conn() as c:
        if game:
            rows = c.execute(f"SELECT * FROM runs WHERE game={ph} ORDER BY created_utc", (game,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM runs ORDER BY created_utc").fetchall()
        out = []
        for row in rows:
            r = dict(row)
            if r["status"] not in OPEN_RUN_STATUSES:
                continue
            signed = c.execute(
                f"SELECT COUNT(*) n FROM run_signups WHERE run_id={ph} "
                f"AND status IN ('waiting','ready_required','ready','active')",
                (r["id"],),
            ).fetchone()["n"]
            if signed >= int(r["players"]):
                continue
            out.append({
                "run_id": r["id"],
                "game": r["game"],
                "status": r["status"],
                "players_needed": int(r["players"]),
                "players_signed_up": int(signed),
                "games": int(r["n_games"]),
                "waiting_expires_at": _utc_after(600),
            })
        return out


def register_agent(display_name: str, token_hash: str, protocol_version: str, sdk_version: str | None = None,
                   agent_id: str | None = None, declared_model: str | None = None,
                   declared_harness: str | None = None) -> dict:
    now, ph = _utcnow(), _ph()
    agent_id = agent_id or f"agent_{uuid.uuid4().hex[:16]}"
    with conn() as c:
        c.execute(
            f"INSERT INTO agents (id,display_name,token_hash,protocol_version,sdk_version,created_utc,last_seen_utc,status,declared_model,declared_harness) "
            f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})",
            (agent_id, display_name, token_hash, protocol_version, sdk_version, now, now, "idle",
             declared_model, declared_harness),
        )
    return get_agent(agent_id)


def list_agents() -> list[dict]:
    """All registered connected agents, most-recently-seen first. Never returns token_hash to callers
    that serialize this — the API layer projects only the public fields."""
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM agents ORDER BY last_seen_utc DESC NULLS LAST, created_utc DESC").fetchall()
        return [_rowdict(r) for r in rows]


def get_agent(agent_id: str) -> dict | None:
    ph = _ph()
    with conn() as c:
        return _rowdict(c.execute(f"SELECT * FROM agents WHERE id={ph}", (agent_id,)).fetchone())


def get_agent_by_token_hash(token_hash: str) -> dict | None:
    ph = _ph()
    with conn() as c:
        return _rowdict(c.execute(f"SELECT * FROM agents WHERE token_hash={ph}", (token_hash,)).fetchone())


def touch_agent(agent_id: str, status: str = "idle") -> None:
    now, ph = _utcnow(), _ph()
    with conn() as c:
        c.execute(f"UPDATE agents SET last_seen_utc={ph}, status={ph} WHERE id={ph}",
                  (now, status, agent_id))


def _active_signups(c, run_id: str) -> list[dict]:
    ph = _ph()
    rows = c.execute(
        f"SELECT s.*, a.display_name FROM run_signups s JOIN agents a ON a.id=s.agent_id "
        f"WHERE s.run_id={ph} AND s.status IN ('waiting','ready_required','ready','active') "
        f"ORDER BY s.created_utc, s.id",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _refresh_run_roster(c, run_id: str, signups: list[dict]) -> None:
    ph = _ph()
    agents = [{"name": s["display_name"], "model": "connected-agent", "harness": "connected",
               "agent_id": s["agent_id"], "signup_id": s["id"]} for s in signups]
    c.execute(f"UPDATE runs SET agents_json={ph} WHERE id={ph}", (json.dumps(agents), run_id))


def _maybe_ready_required(c, run_id: str, ready_deadline_seconds: int = 60) -> None:
    ph = _ph()
    run = c.execute(f"SELECT players,status FROM runs WHERE id={ph}", (run_id,)).fetchone()
    if not run:
        return
    if run["status"] not in OPEN_RUN_STATUSES:
        return
    signups = _active_signups(c, run_id)
    if len(signups) < int(run["players"]):
        c.execute(f"UPDATE runs SET status={ph} WHERE id={ph} AND status!='done'", ("waiting", run_id))
        return
    deadline = _utc_after(ready_deadline_seconds)
    for seat, signup in enumerate(signups[:int(run["players"])]):
        if signup["status"] == "waiting":
            c.execute(
                f"UPDATE run_signups SET status={ph}, seat={ph}, ready_deadline_utc={ph}, updated_utc={ph} "
                f"WHERE id={ph}",
                ("ready_required", seat, deadline, _utcnow(), signup["id"]),
            )
    c.execute(f"UPDATE runs SET status={ph} WHERE id={ph} AND status!='done'", ("ready_required", run_id))
    _refresh_run_roster(c, run_id, _active_signups(c, run_id)[:int(run["players"])])


def create_signup(run_id: str, agent_id: str, max_concurrent_turns: int = 1,
                  waiting_seconds: int = 600) -> tuple[dict | None, str | None]:
    """Create or return this agent's active signup for a run.

    Returns (signup, error_reason). error_reason is one of run_not_found, run_full, run_not_open.
    """
    now, ph = _utcnow(), _ph()
    with conn() as c:
        run = c.execute(f"SELECT * FROM runs WHERE id={ph}", (run_id,)).fetchone()
        if not run:
            return None, "run_not_found"
        existing = c.execute(
            f"SELECT * FROM run_signups WHERE run_id={ph} AND agent_id={ph} "
            f"AND status NOT IN ('completed','rejected','expired','cancelled')",
            (run_id, agent_id),
        ).fetchone()
        if existing:
            _maybe_ready_required(c, run_id)
            return _rowdict(c.execute(f"SELECT * FROM run_signups WHERE id={ph}", (existing["id"],)).fetchone()), None
        active = _active_signups(c, run_id)
        if len(active) >= int(run["players"]):
            return None, "run_full"
        if run["status"] not in OPEN_RUN_STATUSES:
            return None, "run_not_open"
        signup_id = f"signup_{uuid.uuid4().hex[:16]}"
        c.execute(
            f"INSERT INTO run_signups (id,run_id,agent_id,status,seat,created_utc,updated_utc,"
            f"waiting_expires_utc,ready_deadline_utc,last_poll_utc,last_event_id,max_concurrent_turns) "
            f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})",
            (signup_id, run_id, agent_id, "waiting", None, now, now, _utc_after(waiting_seconds),
             None, None, None, int(max_concurrent_turns)),
        )
        _maybe_ready_required(c, run_id)
        return _rowdict(c.execute(f"SELECT * FROM run_signups WHERE id={ph}", (signup_id,)).fetchone()), None


def _expire_signup_if_needed(c, signup: dict) -> dict:
    ph, now = _ph(), _utcnow()
    status = signup["status"]
    if status == "waiting" and signup.get("waiting_expires_utc") and signup["waiting_expires_utc"] < now:
        status = "expired"
    if status == "ready_required" and signup.get("ready_deadline_utc") and signup["ready_deadline_utc"] < now:
        status = "expired"
    if status != signup["status"]:
        c.execute(f"UPDATE run_signups SET status={ph}, updated_utc={ph} WHERE id={ph}",
                  (status, now, signup["id"]))
        signup = dict(signup)
        signup["status"] = status
        signup["updated_utc"] = now
    return signup


def get_signup(signup_id: str, agent_id: str | None = None) -> dict | None:
    ph = _ph()
    with conn() as c:
        if agent_id:
            row = c.execute(f"SELECT * FROM run_signups WHERE id={ph} AND agent_id={ph}",
                            (signup_id, agent_id)).fetchone()
        else:
            row = c.execute(f"SELECT * FROM run_signups WHERE id={ph}", (signup_id,)).fetchone()
        if not row:
            return None
        _maybe_ready_required(c, row["run_id"])
        if agent_id:
            row = c.execute(f"SELECT * FROM run_signups WHERE id={ph} AND agent_id={ph}",
                            (signup_id, agent_id)).fetchone()
        else:
            row = c.execute(f"SELECT * FROM run_signups WHERE id={ph}", (signup_id,)).fetchone()
        return _expire_signup_if_needed(c, dict(row))


def mark_signup_ready(signup_id: str, agent_id: str) -> tuple[dict | None, str | None]:
    now, ph = _utcnow(), _ph()
    with conn() as c:
        row = c.execute(f"SELECT * FROM run_signups WHERE id={ph} AND agent_id={ph}",
                        (signup_id, agent_id)).fetchone()
        if not row:
            return None, "not_found"
        signup = _expire_signup_if_needed(c, dict(row))
        if signup["status"] not in {"ready_required", "ready", "active"}:
            return signup, "not_ready_required"
        if signup["status"] == "ready_required":
            c.execute(f"UPDATE run_signups SET status={ph}, updated_utc={ph} WHERE id={ph}",
                      ("ready", now, signup_id))
        run_id = signup["run_id"]
        signups = _active_signups(c, run_id)
        all_ready = signups and all(s["status"] in {"ready", "active"} for s in signups)
        run = c.execute(f"SELECT players FROM runs WHERE id={ph}", (run_id,)).fetchone()
        if all_ready and run and len(signups) >= int(run["players"]):
            c.execute(f"UPDATE run_signups SET status={ph}, updated_utc={ph} "
                      f"WHERE run_id={ph} AND status IN ('ready','ready_required')",
                      ("active", now, run_id))
            c.execute(f"UPDATE runs SET status={ph} WHERE id={ph} AND status!='done'", ("running", run_id))
            append_event_tx(c, run_id, "run_status", {"status": "active"}, phase="run")
        return _rowdict(c.execute(f"SELECT * FROM run_signups WHERE id={ph}", (signup_id,)).fetchone()), None


def set_coordinator_url(run_id: str, url: str | None) -> None:
    """Publish (or clear) the per-run coordinator's public URL in Neon so the JS registry's signup
    response can hand it to agents to re-point gameplay. Replaces the Modal-Dict handoff."""
    ph = _ph()
    with conn() as c:
        c.execute(f"UPDATE runs SET coordinator_url={ph} WHERE id={ph}", (url, run_id))


def activate_run_if_ready(run_id: str) -> int:
    """Coordinator-driven, race-free activation. When enough seated signups are ready, promote them
    to active in ONE authoritative write, mark the run running, and emit the active event (once).
    Idempotent — safe to call on every poll. Returns the count of seated+active signups.

    This replaces relying on the agents' concurrent mark_ready() each noticing 'all ready' (a race
    that can miss the promotion and never retry, especially across multiple API servers on Neon)."""
    ph, now = _ph(), _utcnow()
    with conn() as c:
        run = c.execute(f"SELECT players FROM runs WHERE id={ph}", (run_id,)).fetchone()
        if not run:
            return 0
        signups = _active_signups(c, run_id)
        ready = [s for s in signups if s.get("seat") is not None and s["status"] in ("ready", "active")]
        if len(ready) >= int(run["players"]):
            cur = c.execute(
                f"UPDATE run_signups SET status={ph}, updated_utc={ph} "
                f"WHERE run_id={ph} AND status IN ('ready','ready_required')",
                ("active", now, run_id))
            if (cur.rowcount or 0) > 0:
                c.execute(f"UPDATE runs SET status={ph} WHERE id={ph} AND status!='done'",
                          ("running", run_id))
                append_event_tx(c, run_id, "run_status", {"status": "active"}, phase="run")
        return sum(1 for s in _active_signups(c, run_id)
                   if s.get("seat") is not None and s["status"] == "active")


def list_run_signups(run_id: str, statuses: set[str] | None = None) -> list[dict]:
    ph = _ph()
    with conn() as c:
        rows = c.execute(
            f"SELECT s.*, a.display_name, a.status agent_status, a.last_seen_utc "
            f"FROM run_signups s JOIN agents a ON a.id=s.agent_id "
            f"WHERE s.run_id={ph} ORDER BY s.seat IS NULL, s.seat, s.created_utc",
            (run_id,),
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            if statuses is None or d["status"] in statuses:
                out.append(_expire_signup_if_needed(c, d))
        return out


def update_run_signups_status(run_id: str, status: str,
                              from_statuses: set[str] | None = None) -> None:
    now, ph = _utcnow(), _ph()
    with conn() as c:
        if from_statuses:
            placeholders = ",".join([ph] * len(from_statuses))
            c.execute(
                f"UPDATE run_signups SET status={ph}, updated_utc={ph} "
                f"WHERE run_id={ph} AND status IN ({placeholders})",
                (status, now, run_id, *sorted(from_statuses)),
            )
        else:
            c.execute(f"UPDATE run_signups SET status={ph}, updated_utc={ph} WHERE run_id={ph}",
                      (status, now, run_id))


def append_event_tx(c, run_id: str, event_type: str, payload: dict,
                    visibility: str = "public", target_signup_id: str | None = None,
                    game_instance_id: str | None = None, phase: str | None = None) -> dict:
    ph = _ph()
    row = c.execute(f"SELECT COALESCE(MAX(seq),0) + 1 n FROM run_events WHERE run_id={ph}",
                    (run_id,)).fetchone()
    seq = int(row["n"])
    now = _utcnow()
    event_id = f"evt_{uuid.uuid4().hex[:16]}"
    c.execute(
        f"INSERT INTO run_events (id,run_id,game_instance_id,seq,visibility,target_signup_id,phase,type,payload_json,created_utc) "
        f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})",
        (event_id, run_id, game_instance_id, seq, visibility, target_signup_id, phase,
         event_type, json.dumps(payload), now),
    )
    return {"id": event_id, "run_id": run_id, "game_instance_id": game_instance_id,
            "seq": seq, "visibility": visibility, "target_signup_id": target_signup_id,
            "phase": phase, "type": event_type, "payload": payload, "created_utc": now}


def append_event(run_id: str, event_type: str, payload: dict,
                 visibility: str = "public", target_signup_id: str | None = None,
                 game_instance_id: str | None = None, phase: str | None = None) -> dict:
    with conn() as c:
        return append_event_tx(c, run_id, event_type, payload, visibility, target_signup_id,
                               game_instance_id, phase)


def list_events_for_signup(signup_id: str, after_event_id: str | None = None,
                           max_events: int = 50) -> list[dict]:
    ph = _ph()
    with conn() as c:
        signup = c.execute(f"SELECT * FROM run_signups WHERE id={ph}", (signup_id,)).fetchone()
        if not signup:
            return []
        after_seq = 0
        if after_event_id:
            row = c.execute(f"SELECT seq FROM run_events WHERE id={ph} AND run_id={ph}",
                            (after_event_id, signup["run_id"])).fetchone()
            after_seq = int(row["seq"]) if row else 0
        rows = c.execute(
            f"SELECT * FROM run_events WHERE run_id={ph} AND seq>{ph} "
            f"AND (visibility='public' OR target_signup_id={ph}) "
            f"ORDER BY seq LIMIT {int(max_events)}",
            (signup["run_id"], after_seq, signup_id),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json"))
            d["event_id"] = d.pop("id")
            out.append(d)
        if out:
            c.execute(f"UPDATE run_signups SET last_event_id={ph}, last_poll_utc={ph} WHERE id={ph}",
                      (out[-1]["event_id"], _utcnow(), signup_id))
        else:
            c.execute(f"UPDATE run_signups SET last_poll_utc={ph} WHERE id={ph}", (_utcnow(), signup_id))
        return out


def list_run_events(run_id: str, max_events: int = 200) -> list[dict]:
    ph = _ph()
    with conn() as c:
        rows = c.execute(
            f"SELECT * FROM run_events WHERE run_id={ph} ORDER BY seq DESC LIMIT {int(max_events)}",
            (run_id,),
        ).fetchall()
        out = []
        for r in reversed(rows):
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json"))
            d["event_id"] = d.pop("id")
            out.append(d)
        return out


def create_turn(run_id: str, signup_id: str, game_instance_id: str, seat: int, phase: str,
                action_kind: str, observation: dict, legal_action: dict,
                deadline_seconds: int = 60) -> dict:
    now, ph = _utcnow(), _ph()
    turn_id = f"turn_{uuid.uuid4().hex[:16]}"
    with conn() as c:
        c.execute(
            f"INSERT INTO turns (id,run_id,signup_id,game_instance_id,seat,phase,action_kind,"
            f"observation_json,legal_action_json,status,deadline_utc,created_utc,claimed_utc) "
            f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})",
            (turn_id, run_id, signup_id, game_instance_id, seat, phase, action_kind,
             json.dumps(observation), json.dumps(legal_action), "pending",
             _utc_after(deadline_seconds), now, None),
        )
    return get_turn(turn_id)


def get_turn(turn_id: str) -> dict | None:
    ph = _ph()
    with conn() as c:
        row = c.execute(f"SELECT * FROM turns WHERE id={ph}", (turn_id,)).fetchone()
        return _json_row(row, "observation_json", "legal_action_json")


def get_turn_reply(turn_id: str) -> dict | None:
    ph = _ph()
    with conn() as c:
        row = c.execute(f"SELECT * FROM turn_replies WHERE turn_id={ph}", (turn_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["action"] = json.loads(d.pop("action_json"))
        return d


def pending_turn_for_signup(signup_id: str) -> dict | None:
    ph = _ph()
    with conn() as c:
        row = c.execute(
            f"SELECT * FROM turns WHERE signup_id={ph} AND status='pending' "
            f"ORDER BY created_utc LIMIT 1",
            (signup_id,),
        ).fetchone()
        if not row:
            return None
        now = _utcnow()
        c.execute(f"UPDATE turns SET claimed_utc=COALESCE(claimed_utc,{ph}) WHERE id={ph}",
                  (now, row["id"]))
        row = c.execute(f"SELECT * FROM turns WHERE id={ph}", (row["id"],)).fetchone()
        return _json_row(row, "observation_json", "legal_action_json")


def reply_to_turn(turn_id: str, agent_id: str, action: dict, reasoning: str | None,
                  client_ms: int | None = None) -> tuple[dict | None, str | None]:
    now, ph = _utcnow(), _ph()
    with conn() as c:
        turn = c.execute(
            f"SELECT t.*, s.agent_id FROM turns t JOIN run_signups s ON s.id=t.signup_id "
            f"WHERE t.id={ph}",
            (turn_id,),
        ).fetchone()
        if not turn:
            return None, "not_found"
        if turn["agent_id"] != agent_id:
            return None, "forbidden"
        existing = c.execute(f"SELECT * FROM turn_replies WHERE turn_id={ph}", (turn_id,)).fetchone()
        if existing:
            return _rowdict(existing), "reply_already_recorded"
        if turn["deadline_utc"] and turn["deadline_utc"] < now:
            return None, "turn_expired"
        legal_action = json.loads(turn["legal_action_json"]) if turn["legal_action_json"] else {}
        ok, err = _validate_action(action, legal_action)
        if not ok:
            return {"reason": err}, "invalid_action"
        c.execute(
            f"INSERT INTO turn_replies (turn_id,signup_id,action_json,reasoning,client_ms,accepted,created_utc) "
            f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph})",
            (turn_id, turn["signup_id"], json.dumps(action), reasoning, client_ms, 1, now),
        )
        c.execute(f"UPDATE turns SET status={ph} WHERE id={ph}", ("replied", turn_id))
        return _rowdict(c.execute(f"SELECT * FROM turn_replies WHERE turn_id={ph}", (turn_id,)).fetchone()), None


def default_turn(turn_id: str, action: dict, reasoning: str = "(deadline expired; defaulted)") -> dict | None:
    now, ph = _utcnow(), _ph()
    with conn() as c:
        turn = c.execute(f"SELECT * FROM turns WHERE id={ph}", (turn_id,)).fetchone()
        if not turn:
            return None
        existing = c.execute(f"SELECT * FROM turn_replies WHERE turn_id={ph}", (turn_id,)).fetchone()
        if existing:
            return _rowdict(existing)
        c.execute(
            f"INSERT INTO turn_replies (turn_id,signup_id,action_json,reasoning,client_ms,accepted,created_utc) "
            f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph})",
            (turn_id, turn["signup_id"], json.dumps(action), reasoning, None, 0, now),
        )
        c.execute(f"UPDATE turns SET status={ph} WHERE id={ph}", ("defaulted", turn_id))
        return _rowdict(c.execute(f"SELECT * FROM turn_replies WHERE turn_id={ph}", (turn_id,)).fetchone())


def list_run_turn_debug(run_id: str, max_turns: int = 100) -> list[dict]:
    ph = _ph()
    with conn() as c:
        rows = c.execute(
            f"SELECT t.*, r.action_json, r.reasoning, r.client_ms, r.accepted, r.created_utc reply_created_utc "
            f"FROM turns t LEFT JOIN turn_replies r ON r.turn_id=t.id "
            f"WHERE t.run_id={ph} ORDER BY t.created_utc DESC LIMIT {int(max_turns)}",
            (run_id,),
        ).fetchall()
        out = []
        for row in rows:
            d = _json_row(row, "observation_json", "legal_action_json")
            if d.get("action_json"):
                d["reply_action"] = json.loads(d.pop("action_json"))
            out.append(d)
        return out
