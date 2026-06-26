"""VDD (FINDINGS #2/#3 codex round-7): explicit-seat uniqueness must be RACE-FREE.

The pre-insert duplicate-seat read in store.create_signup / the JS signup handler is a non-atomic
TOCTOU: two concurrent signups requesting the SAME explicit seat can both pass the duplicate check,
then both INSERT the same roster_index (run_signups had no uniqueness constraint on
(run_id, roster_index)). advanceLobby / _maybe_ready_required would then assign duplicate
deterministic seats, corrupting the identity->seat contract (SPEC D5/V-7).

FIX: a PARTIAL UNIQUE INDEX uq_run_signups_run_roster ON run_signups(run_id, roster_index)
WHERE roster_index IS NOT NULL. NULL roster_index (normal/no-seat signups) is excluded by the
partial WHERE, so many NULLs are allowed (INV-2 holds). create_signup additionally catches the
unique-constraint violation around the INSERT and returns 'invalid_seat' so even a racing insert is
rejected atomically.

RED before the fix: no index exists, so two rows with the same (run_id, roster_index) insert fine.
"""
from __future__ import annotations

import sqlite3

import pytest

from arena import store


@pytest.fixture()
def _local_db(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "seat_uq.db")
    return tmp_path


def test_partial_unique_index_exists(_local_db):
    """The partial unique index on (run_id, roster_index) WHERE roster_index IS NOT NULL exists."""
    with store.conn() as c:
        row = c.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='uq_run_signups_run_roster'"
        ).fetchone()
    assert row is not None, "expected partial unique index uq_run_signups_run_roster to exist"
    sql = (row["sql"] or "").lower()
    assert "run_id" in sql and "roster_index" in sql
    assert "where" in sql and "roster_index is not null" in sql, (
        "index must be PARTIAL (WHERE roster_index IS NOT NULL) so NULLs are unconstrained (INV-2)"
    )


def test_duplicate_roster_index_insert_raises_integrity_error(_local_db):
    """A direct second INSERT of the same (run_id, roster_index) raises IntegrityError (atomic guard)."""
    with store.conn() as c:
        c.execute(
            "INSERT INTO run_signups (id,run_id,agent_id,status,roster_index) VALUES (?,?,?,?,?)",
            ("s1", "run_uq", "a1", "waiting", 0),
        )
        with pytest.raises(sqlite3.IntegrityError):
            c.execute(
                "INSERT INTO run_signups (id,run_id,agent_id,status,roster_index) VALUES (?,?,?,?,?)",
                ("s2", "run_uq", "a2", "waiting", 0),
            )


def test_many_null_roster_index_rows_allowed(_local_db):
    """INV-2: the partial index excludes NULL roster_index, so many no-seat signups coexist."""
    with store.conn() as c:
        for i in range(4):
            c.execute(
                "INSERT INTO run_signups (id,run_id,agent_id,status,roster_index) VALUES (?,?,?,?,?)",
                (f"n{i}", "run_null", f"a{i}", "waiting", None),
            )
        n = c.execute(
            "SELECT COUNT(*) c FROM run_signups WHERE run_id='run_null'"
        ).fetchone()["c"]
    assert n == 4


def test_create_signup_racing_insert_rejected_invalid_seat(_local_db, monkeypatch):
    """Even if the friendly pre-check is bypassed, a colliding INSERT is caught and returns invalid_seat.

    We simulate a TOCTOU race by inserting a competing row at the target roster_index AFTER the
    pre-read has happened but BEFORE create_signup's own INSERT, by monkeypatching the duplicate-seat
    pre-check to a no-op. The DB unique index must then reject the racing insert atomically.
    """
    # Seed a run that is open and joinable.
    store.create_connected_run({
        "id": "run_race", "game": "onuw", "label": "race", "status": "open",
        "n_games": 1, "players": 5, "seed_base": 1,
    })
    store.register_agent("racer-a", "hash-a", "arena-agent-v1", agent_id="agent_a")
    store.register_agent("racer-b", "hash-b", "arena-agent-v1", agent_id="agent_b")

    # First agent legitimately takes seat 2.
    sg, err = store.create_signup("run_race", "agent_a", seat=2)
    assert err is None, err

    # The DB partial unique index must reject the duplicate seat 2 atomically -> invalid_seat,
    # independent of the friendly pre-read (which would also catch it; the index is the race-free arbiter).
    sg2, err2 = store.create_signup("run_race", "agent_b", seat=2)
    assert sg2 is None
    assert err2 == "invalid_seat", f"racing duplicate seat must return invalid_seat, got {err2!r}"


def test_create_signup_friendly_duplicate_still_invalid_seat(_local_db):
    """The friendly pre-check path still returns invalid_seat for an already-taken seat."""
    store.create_connected_run({
        "id": "run_friendly", "game": "onuw", "label": "f", "status": "open",
        "n_games": 1, "players": 5, "seed_base": 1,
    })
    store.register_agent("fa", "h-fa", "arena-agent-v1", agent_id="agent_fa")
    store.register_agent("fb", "h-fb", "arena-agent-v1", agent_id="agent_fb")
    sg, err = store.create_signup("run_friendly", "agent_fa", seat=1)
    assert err is None, err
    sg2, err2 = store.create_signup("run_friendly", "agent_fb", seat=1)
    assert sg2 is None and err2 == "invalid_seat"
