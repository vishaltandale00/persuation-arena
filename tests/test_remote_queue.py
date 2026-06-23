from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

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


def _agents():
    return [{"name": f"A{i}", "model": f"m{i}", "harness": "base"} for i in range(5)]


def _job(owner="alice", run_id="r1", job_id="j1"):
    return {
        "id": job_id,
        "run_id": run_id,
        "owner": owner,
        "game": "onuw",
        "label": "ONUW",
        "n_games": 2,
        "players": 5,
        "seed_base": 9000,
        "rounds": 1,
        "agents": _agents(),
    }


def _transcript(gid=1):
    return {
        "seed": 9000,
        "winner_team": "good",
        "outcome": {"text": f"game {gid}", "team": "good"},
        "players": [
            {"seat": i, "dealt": "Villager", "end": "Villager", "team": "good",
             "won": True, "calls": 1, "forfeits": 0}
            for i in range(5)
        ],
    }


def test_claim_is_owner_scoped_and_single_winner(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "arena.db")

    store.enqueue_job(_job(owner="alice", run_id="ra", job_id="ja"))
    store.enqueue_job(_job(owner="bob", run_id="rb", job_id="jb"))

    bob = store.claim_job("bob", "w1", lease_seconds=60)
    assert bob["id"] == "jb"
    assert store.claim_job("bob", "w2", lease_seconds=60) is None

    alice = store.claim_job("alice", "w3", lease_seconds=60)
    assert alice["id"] == "ja"


def test_expired_lease_reclaims_same_job(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "arena.db")

    store.enqueue_job(_job())
    first = store.claim_job("alice", "w1", lease_seconds=-1)
    assert first["worker_id"] == "w1"

    reclaimed = store.claim_job("alice", "w2", lease_seconds=60)
    assert reclaimed["id"] == first["id"]
    assert reclaimed["worker_id"] == "w2"


def test_heartbeat_and_finish_are_worker_fenced(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "arena.db")

    store.enqueue_job(_job())
    store.claim_job("alice", "w1", lease_seconds=60)

    assert store.heartbeat_job("j1", "other", lease_seconds=60) is False
    assert store.heartbeat_job("j1", "w1", lease_seconds=60) is True
    assert store.finish_job("j1", "other", "done") is False
    assert store.finish_job("j1", "w1", "done") is True
    assert store.get_job("j1")["status"] == "done"
    assert store.get_run("r1")["status"] == "done"


def test_api_submit_claim_ingest_complete(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("INGEST_TOKENS", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "arena.db")

    from arena.server import app

    with TestClient(app) as client:
        submitted = client.post("/api/runs", json={
            "owner": "alice", "game": "onuw", "games": 1, "rounds": 1,
            "agents": _agents(),
        })
        assert submitted.status_code == 200
        run_id = submitted.json()["run_id"]
        job_id = submitted.json()["job_id"]

        claimed = client.post("/api/jobs/claim", json={"owner": "alice", "worker_id": "w1"})
        assert claimed.status_code == 200
        assert claimed.json()["job"]["id"] == job_id

        ingested = client.post("/api/ingest", json={
            "owner": "alice", "job_id": job_id, "run_id": run_id, "gid": 1,
            "transcript": _transcript(1), "agents": _agents(),
        })
        assert ingested.status_code == 200
        assert ingested.json()["inserted"] is True

        duplicate = client.post("/api/ingest", json={
            "owner": "alice", "job_id": job_id, "run_id": run_id, "gid": 1,
            "transcript": _transcript(1), "agents": _agents(),
        })
        assert duplicate.status_code == 200
        assert duplicate.json()["inserted"] is False

        completed = client.post("/api/jobs/complete", json={
            "owner": "alice", "job_id": job_id, "worker_id": "w1", "status": "done",
        })
        assert completed.status_code == 200

        run = client.get(f"/api/runs/{run_id}").json()
        assert run["status"] == "done"
        assert len(run["games"]) == 1


def test_api_worker_token_owner_fence(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("INGEST_TOKENS", "alice=secret")
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "arena.db")

    from arena.server import app

    with TestClient(app) as client:
        client.post("/api/runs", json={"owner": "alice", "game": "onuw", "games": 1,
                                       "agents": _agents()}).raise_for_status()
        no_token = client.post("/api/jobs/claim", json={"owner": "alice", "worker_id": "w1"})
        assert no_token.status_code == 401

        wrong_owner = client.post(
            "/api/jobs/claim",
            headers={"Authorization": "Bearer secret"},
            json={"owner": "bob", "worker_id": "w1"},
        )
        assert wrong_owner.status_code == 403

        ok = client.post(
            "/api/jobs/claim",
            headers={"Authorization": "Bearer secret"},
            json={"owner": "alice", "worker_id": "w1"},
        )
        assert ok.status_code == 200
        assert ok.json()["job"]["owner"] == "alice"


@pytest.mark.skipif(not PG_OK, reason="needs Postgres to exercise SKIP LOCKED claim SQL")
def test_postgres_claim_is_owner_scoped_and_single_winner(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", PG_URL)
    store.init_schema()
    with store.conn() as c:
        c.execute("TRUNCATE jobs, game_players, games, runs")

    store.enqueue_job(_job(owner="alice", run_id="pg_ra", job_id="pg_ja"))
    store.enqueue_job(_job(owner="bob", run_id="pg_rb", job_id="pg_jb"))

    bob = store.claim_job("bob", "w1", lease_seconds=60)
    assert bob["id"] == "pg_jb"
    assert store.claim_job("bob", "w2", lease_seconds=60) is None

    alice = store.claim_job("alice", "w3", lease_seconds=60)
    assert alice["id"] == "pg_ja"
