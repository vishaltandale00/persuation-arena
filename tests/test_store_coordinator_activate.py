from __future__ import annotations

from arena import store


def _sqlite_store(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store.init_schema()


def _make_run(run_id: str, players: int = 2):
    store.create_connected_run({
        "id": run_id, "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 1, "players": players, "seed_base": 1,
    })


def _signup(run_id: str, agent_name: str) -> tuple[str, str]:
    """Public-path signup mirroring tests/test_connected_runner.py _active_signup."""
    agent = store.register_agent(agent_name, f"hash_{agent_name}", "arena-agent-v1", "test")
    signup, err = store.create_signup(run_id, agent["id"])
    assert err is None, err
    return agent["id"], signup["id"]


def _activation_events(run_id: str) -> list[dict]:
    """run_status events carrying status=='active' (the activation signal)."""
    return [
        e for e in store.list_run_events(run_id, max_events=1000)
        if e["type"] == "run_status" and e["payload"].get("status") == "active"
    ]


# --- set_coordinator_url ---------------------------------------------------------


def test_set_coordinator_url_round_trips(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    _make_run("run_url")
    assert store.get_run("run_url")["coordinator_url"] is None

    store.set_coordinator_url("run_url", "https://coord.example/run_url")
    assert store.get_run("run_url")["coordinator_url"] == "https://coord.example/run_url"


def test_set_coordinator_url_none_clears(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    _make_run("run_url_clear")
    store.set_coordinator_url("run_url_clear", "https://coord.example/x")
    assert store.get_run("run_url_clear")["coordinator_url"] == "https://coord.example/x"

    store.set_coordinator_url("run_url_clear", None)
    assert store.get_run("run_url_clear")["coordinator_url"] is None


# --- activate_run_if_ready -------------------------------------------------------


def test_activate_nonexistent_run_returns_zero(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    assert store.activate_run_if_ready("nope") == 0


def test_activate_with_not_all_ready_returns_zero_and_status_unchanged(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    _make_run("run_partial", players=2)
    a0, s0 = _signup("run_partial", "p0")
    a1, s1 = _signup("run_partial", "p1")
    # Both signups now seated + ready_required (the run filled). Mark only ONE ready.
    ready, err = store.mark_signup_ready(s0, a0)
    assert err is None or err == "not_ready_required"
    prior_status = store.get_run("run_partial")["status"]
    assert prior_status == "ready_required"

    assert store.activate_run_if_ready("run_partial") == 0
    assert store.get_run("run_partial")["status"] == prior_status
    assert _activation_events("run_partial") == []


def test_activate_with_all_ready_activates_signups_and_run(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    _make_run("run_ready", players=3)
    pairs = [_signup("run_ready", f"p{i}") for i in range(3)]
    for agent_id, signup_id in pairs:
        ready, err = store.mark_signup_ready(signup_id, agent_id)
        assert err is None or err == "not_ready_required"

    n = store.activate_run_if_ready("run_ready")
    assert n == 3
    assert store.get_run("run_ready")["status"] == "running"
    assert {s["status"] for s in store.list_run_signups("run_ready")} == {"active"}
    assert len(_activation_events("run_ready")) == 1


def test_activate_is_idempotent_single_active_event(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    _make_run("run_idem", players=2)
    pairs = [_signup("run_idem", f"p{i}") for i in range(2)]
    for agent_id, signup_id in pairs:
        ready, err = store.mark_signup_ready(signup_id, agent_id)
        assert err is None or err == "not_ready_required"

    first = store.activate_run_if_ready("run_idem")
    second = store.activate_run_if_ready("run_idem")
    third = store.activate_run_if_ready("run_idem")
    assert first == second == third == 2
    assert store.get_run("run_idem")["status"] == "running"
    assert len(_activation_events("run_idem")) == 1
