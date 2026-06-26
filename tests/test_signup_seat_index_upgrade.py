"""VDD (FINDING P1 codex round-8 + FINDING P2): the partial unique index
uq_run_signups_run_roster must (1) be created AFTER the roster_index column migration so an
EXISTING pre-upgrade DB can migrate without "no such column: roster_index", and (2) carry an
ACTIVE-STATUS predicate so an expired/cancelled signup releases its seat for a replacement.

P1 (UPGRADE PATH): an old SQLite DB whose run_signups predates roster_index must open + migrate
cleanly. RED before the fix: the index DDL ran inside executescript(SQLITE_SCHEMA) BEFORE _migrate
added the column, so on a pre-upgrade table the index creation raised OperationalError
("no such column: roster_index"), blocking startup/migration (and prod Neon migrate-neon.yml).

P2 (EXPIRED-SEAT FREES UP): the index covered EVERY non-null roster_index, so an expired/cancelled
signup kept "holding" its seat and a replacement agent for that seat hit the unique violation and got
'invalid_seat' forever, wedging the shard. FIX: only live holders count
(status IN ('waiting','ready_required','ready','active')).
"""
from __future__ import annotations

import sqlite3

import pytest

from arena import store


@pytest.fixture()
def _local_db(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "seat_upgrade.db")
    return tmp_path


# Active-status set that create_signup treats as "occupying" a seat (mirror exactly).
_ACTIVE = ("waiting", "ready_required", "ready", "active")


def _build_pre_upgrade_db(db_path) -> None:
    """Create a run_signups table WITHOUT roster_index, simulating a DB that predates that column.

    Critically this must NOT define uq_run_signups_run_roster (a pre-upgrade DB never had it), so
    that the migration path is the thing under test.
    """
    c = sqlite3.connect(db_path)
    try:
        c.execute(
            "CREATE TABLE run_signups ("
            "  id TEXT PRIMARY KEY, run_id TEXT, agent_id TEXT, status TEXT, seat INTEGER,"
            "  created_utc TEXT, updated_utc TEXT, waiting_expires_utc TEXT, ready_deadline_utc TEXT,"
            "  last_poll_utc TEXT, last_event_id TEXT, max_concurrent_turns INTEGER DEFAULT 1,"
            "  UNIQUE (run_id, agent_id)"
            ")"
        )
        # A representative old row with NO roster_index column at all.
        c.execute(
            "INSERT INTO run_signups (id,run_id,agent_id,status) VALUES (?,?,?,?)",
            ("old1", "run_old", "agent_old", "waiting"),
        )
        c.commit()
    finally:
        c.close()


def test_upgrade_from_pre_roster_index_db_succeeds(_local_db):
    """P1: opening an old DB (no roster_index col) migrates cleanly and the index exists afterward.

    RED before the fix: conn() -> executescript(SQLITE_SCHEMA) tried to CREATE the partial index on a
    table that still lacked roster_index, raising OperationalError "no such column: roster_index".
    """
    db_path = _local_db / "seat_upgrade.db"
    _build_pre_upgrade_db(db_path)

    # This must NOT raise. The index is created only after _migrate adds roster_index.
    with store.conn() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(run_signups)").fetchall()}
        assert "roster_index" in cols, "migration must add roster_index"
        idx = c.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='uq_run_signups_run_roster'"
        ).fetchone()
        assert idx is not None, "partial unique index must exist after migration"


def test_index_has_active_status_predicate(_local_db):
    """P2: the index WHERE clause restricts to live (active-status) holders, not every non-null seat."""
    with store.conn() as c:
        row = c.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='uq_run_signups_run_roster'"
        ).fetchone()
    assert row is not None
    sql = (row["sql"] or "").lower()
    assert "roster_index is not null" in sql
    for st in _ACTIVE:
        assert st in sql, f"index predicate must include active status {st!r}; got: {sql}"


def test_expired_signup_frees_seat_for_replacement(_local_db):
    """P2: an expired holder releases its seat so a different agent can take seat k (not invalid_seat)."""
    store.create_connected_run({
        "id": "run_free", "game": "onuw", "label": "free", "status": "open",
        "n_games": 1, "players": 5, "seed_base": 1,
    })
    store.register_agent("a", "h-a", "arena-agent-v1", agent_id="agent_a")
    store.register_agent("b", "h-b", "arena-agent-v1", agent_id="agent_b")

    sg, err = store.create_signup("run_free", "agent_a", seat=3)
    assert err is None, err

    # Agent A's seat-3 signup expires (e.g. waiting window elapsed).
    with store.conn() as c:
        c.execute(
            "UPDATE run_signups SET status='expired' WHERE id=?", (sg["id"],),
        )

    # A DIFFERENT agent must now be able to take seat 3 — the expired row no longer holds it.
    sg2, err2 = store.create_signup("run_free", "agent_b", seat=3)
    assert err2 is None, f"expired seat must be reclaimable, got {err2!r}"
    assert sg2 is not None


def test_two_active_signups_same_seat_still_rejected(_local_db):
    """INV: two ACTIVE signups on the same seat are still rejected at the DB level (atomic guard)."""
    with store.conn() as c:
        c.execute(
            "INSERT INTO run_signups (id,run_id,agent_id,status,roster_index) VALUES (?,?,?,?,?)",
            ("s1", "run_act", "a1", "active", 0),
        )
        with pytest.raises(sqlite3.IntegrityError):
            c.execute(
                "INSERT INTO run_signups (id,run_id,agent_id,status,roster_index) VALUES (?,?,?,?,?)",
                ("s2", "run_act", "a2", "waiting", 0),
            )


def test_expired_and_active_same_seat_coexist(_local_db):
    """P2: an expired holder and a live holder of the same seat can coexist (predicate excludes expired)."""
    with store.conn() as c:
        c.execute(
            "INSERT INTO run_signups (id,run_id,agent_id,status,roster_index) VALUES (?,?,?,?,?)",
            ("e1", "run_mix", "a1", "expired", 0),
        )
        # A live holder of the same seat must be insertable (expired one is excluded by the predicate).
        c.execute(
            "INSERT INTO run_signups (id,run_id,agent_id,status,roster_index) VALUES (?,?,?,?,?)",
            ("a1", "run_mix", "a2", "waiting", 0),
        )
        n = c.execute(
            "SELECT COUNT(*) c FROM run_signups WHERE run_id='run_mix' AND roster_index=0"
        ).fetchone()["c"]
    assert n == 2
