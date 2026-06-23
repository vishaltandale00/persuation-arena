"""Dual-backend store tests: run against SQLite (always) AND a real Postgres (when reachable),
so the federated correctness properties — first-writer-wins games and monotonic run status — are
verified on both backends, and the two backends produce identical query output for the same input.

Point the Postgres backend at a test DB via ARENA_TEST_DATABASE_URL (defaults to the local Docker
container on :5433). When no Postgres is reachable the 'pg' parametrization is skipped, so the
normal `pytest tests/` run (SQLite only) is unaffected.
"""
from __future__ import annotations

import os

import pytest

from arena import store

PG_URL = os.environ.get("ARENA_TEST_DATABASE_URL", "postgresql://postgres:arena@localhost:5433/arena")


def _pg_available() -> bool:
    try:
        import psycopg
        psycopg.connect(PG_URL, connect_timeout=2).close()
        return True
    except Exception:
        return False


PG_OK = _pg_available()


@pytest.fixture(params=["sqlite", "pg"])
def backend(request, tmp_path, monkeypatch):
    if request.param == "pg":
        if not PG_OK:
            pytest.skip("no test Postgres (start docker pg or set ARENA_TEST_DATABASE_URL)")
        monkeypatch.setenv("DATABASE_URL", PG_URL)
        store.init_schema()
        with store.conn() as c:
            for t in ("game_players", "games", "runs"):
                c.execute(f"TRUNCATE {t}")
        yield "pg"
    else:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
        yield "sqlite"


# --- fixtures of data --------------------------------------------------------
def _run_meta(rid="r1", status="running", submitter=None, created_utc=None):
    return {
        "id": rid, "game": "onuw", "label": "ONUW", "status": status,
        "n_games": 2, "players": 5, "seed_base": 9000, "created": "2026-06-23 09:00",
        "agents": [{"name": f"A{i}", "model": f"m{i}", "harness": "base"} for i in range(5)],
        "submitter": submitter, "created_utc": created_utc,
    }


def _transcript(seed, winner, won_seat0=True, line="ok"):
    players = [
        {"seat": i, "dealt": "Villager", "end": "Villager",
         "team": ("evil" if (i == 0 and winner == "evil") else "good"),
         "won": (i == 0 and won_seat0), "calls": 3, "forfeits": 0}
        for i in range(5)
    ]
    return {"seed": seed, "winner_team": winner, "outcome": {"text": line, "team": winner},
            "players": players}


def _agents():
    return [{"name": f"A{i}", "model": f"m{i}"} for i in range(5)]


# --- tests -------------------------------------------------------------------
def test_save_and_get_run(backend):
    store.save_run(_run_meta())
    store.save_game("r1", 1, _transcript(9000, "good"), _agents())
    r = store.get_run("r1")
    assert r is not None and r["status"] == "running"
    assert len(r["games"]) == 1 and r["games"][0]["gid"] == 1 and r["games"][0]["winner_team"] == "good"
    assert store.distinct_gids("r1") == [1]
    assert store.get_game("r1", 1)["winner_team"] == "good"
    assert len(store.player_rows("r1")) == 5


def test_first_writer_wins_game(backend):
    """A re-published (run_id,gid) must NOT overwrite the recorded outcome (LLM games aren't
    reproducible, so a reclaim's re-run is a different valid game keyed to the same gid)."""
    store.save_run(_run_meta())
    store.save_game("r1", 1, _transcript(9000, "good", line="first"), _agents())
    store.save_game("r1", 1, _transcript(9000, "evil", line="second"), _agents())  # must be ignored
    g = store.get_game("r1", 1)
    assert g["winner_team"] == "good" and g["outcome"]["text"] == "first"
    assert len(store.player_rows("r1")) == 5  # not duplicated by the second write


def test_monotonic_save_run_never_regresses_done(backend):
    store.save_run(_run_meta(status="running"))
    store.update_run_status("r1", "done")
    store.save_run(_run_meta(status="running"))      # late/duplicate per-game re-publish
    assert store.get_run("r1")["status"] == "done"
    store.update_run_status("r1", "running")          # stray late status write
    assert store.get_run("r1")["status"] == "done"


def test_partial_can_upgrade_to_done(backend):
    store.save_run(_run_meta(status="running"))
    store.update_run_status("r1", "partial")
    assert store.get_run("r1")["status"] == "partial"
    store.update_run_status("r1", "done")             # a resume completes the run
    assert store.get_run("r1")["status"] == "done"


def test_list_runs_ordered_by_created_utc(backend):
    store.save_run(_run_meta(rid="old", created_utc="2026-06-23T09:00:00Z"))
    store.save_run(_run_meta(rid="new", created_utc="2026-06-23T10:00:00Z"))
    ids = [r["id"] for r in store.list_runs()]
    assert ids.index("new") < ids.index("old")


def test_team_split_and_wins(backend):
    store.save_run(_run_meta())
    store.save_game("r1", 1, _transcript(9000, "good", won_seat0=True), _agents())
    store.save_game("r1", 2, _transcript(9001, "evil", won_seat0=True), _agents())
    r = store.get_run("r1")
    assert r["team_split"] == {"good": 1, "evil": 1}
    assert r["wins"]["A0"] == 2  # seat-0 agent won both


@pytest.mark.skipif(not PG_OK, reason="needs Postgres to compare backends")
def test_sqlite_pg_parity(tmp_path, monkeypatch):
    """Same inputs through both backends must yield identical get_run output (sans nothing)."""
    def run_on_sqlite():
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setattr(store, "DB_PATH", tmp_path / "parity.db")
        store.save_run(_run_meta(rid="p", submitter="alice", created_utc="2026-06-23T09:00:00Z"))
        store.save_game("p", 1, _transcript(9000, "good"), _agents())
        store.save_game("p", 2, _transcript(9001, "evil"), _agents())
        return store.get_run("p")

    def run_on_pg():
        monkeypatch.setenv("DATABASE_URL", PG_URL)
        store.init_schema()
        with store.conn() as c:
            for t in ("game_players", "games", "runs"):
                c.execute(f"TRUNCATE {t}")
        store.save_run(_run_meta(rid="p", submitter="alice", created_utc="2026-06-23T09:00:00Z"))
        store.save_game("p", 1, _transcript(9000, "good"), _agents())
        store.save_game("p", 2, _transcript(9001, "evil"), _agents())
        return store.get_run("p")

    a = run_on_sqlite()
    b = run_on_pg()
    assert a == b
