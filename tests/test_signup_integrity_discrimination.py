"""Codex round-9: store.create_signup's unique-violation handler must map ONLY the explicit-seat
index (uq_run_signups_run_roster) to 'invalid_seat'. A (run_id, agent_id) collision (a racing
duplicate, or this agent's prior/expired signup) must reload the existing signup, not surface a
misleading seat error (the broad except was a regression affecting normal no-seat runs)."""
from __future__ import annotations

import sqlite3

from arena import store


def _sqlite_store(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "sig.db")
    store.init_schema()


def test_is_seat_index_violation_discriminates():
    """Seat index vs the (run_id, agent_id) constraint, by the name in the error message — both
    SQLite and Postgres phrasings."""
    seat_sqlite = sqlite3.IntegrityError(
        "UNIQUE constraint failed: run_signups.run_id, run_signups.roster_index")
    agent_sqlite = sqlite3.IntegrityError(
        "UNIQUE constraint failed: run_signups.run_id, run_signups.agent_id")
    seat_pg = Exception('duplicate key value violates unique constraint "uq_run_signups_run_roster"')
    agent_pg = Exception('duplicate key value violates unique constraint "run_signups_run_id_agent_id_key"')
    assert store._is_seat_index_violation(seat_sqlite) is True
    assert store._is_seat_index_violation(agent_sqlite) is False
    assert store._is_seat_index_violation(seat_pg) is True
    assert store._is_seat_index_violation(agent_pg) is False


def test_agent_id_collision_reloads_not_invalid_seat(tmp_path, monkeypatch):
    """A (run_id, agent_id) INSERT collision must reload the existing signup, NOT report
    'invalid_seat'. Deterministically triggered: expiring the agent's signup makes create_signup's
    existence pre-check (which excludes expired) miss, so it proceeds to INSERT and hits the
    UNIQUE(run_id, agent_id) constraint the expired row still occupies. Before the fix the broad
    except mislabeled this 'invalid_seat'; the fix reloads the existing (expired) row."""
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "agc", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 1, "players": 5, "seed_base": 1})
    agent = store.register_agent("agx", "hash_agx", "arena-agent-v1", "test")

    s1, err = store.create_signup("agc", agent["id"])  # no seat -> normal path
    assert err is None and s1 is not None

    # Force the existence pre-check to miss (status excluded) while the row still occupies (run,agent).
    with store.conn() as c:
        c.execute("UPDATE run_signups SET status='expired' WHERE id=?", (s1["id"],))

    s2, err2 = store.create_signup("agc", agent["id"])
    assert err2 != "invalid_seat", "a (run_id, agent_id) collision must NOT be reported as a seat error"
    assert err2 is None and s2 is not None
    assert s2["id"] == s1["id"], "must reload the existing signup row, not create/return a new one"
