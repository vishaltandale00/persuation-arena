"""P0 — identity linkage: game_players carries agent_id/signup_id, agents carry declared model/harness.

These are the prerequisite for cross-run rating keyed by bearer-token identity. The connected
end-to-end test is the real acceptance: a connected ONUW game must write game_players rows whose
agent_id matches the registered agents (the value flows through save_game's meta).
"""
from __future__ import annotations

import sqlite3
import threading
import time

from arena import store
from arena.identity import NO_ONE_REF


def _sqlite(tmp_path, monkeypatch, name="t.db"):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / name)
    store.init_schema()


def _transcript(winner="good"):
    return {
        "seed": 1, "winner_team": winner, "outcome": {"text": "village wins"},
        "players": [
            {"seat": 0, "dealt": "Villager", "end": "Villager", "team": "good", "won": True,
             "calls": 2, "forfeits": 0},
            {"seat": 1, "dealt": "Werewolf", "end": "Werewolf", "team": "evil", "won": False,
             "calls": 2, "forfeits": 1},
        ],
    }


def test_fresh_schema_has_identity_columns(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    with store.conn() as c:
        gp = {r["name"] for r in c.execute("PRAGMA table_info(game_players)").fetchall()}
        ag = {r["name"] for r in c.execute("PRAGMA table_info(agents)").fetchall()}
        tables = {r["name"] for r in
                  c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"agent_id", "signup_id"} <= gp
    assert {"declared_model", "declared_harness"} <= ag
    assert {"ratings", "rating_events", "role_difficulty"} <= tables


def test_migration_upgrades_legacy_db(tmp_path, monkeypatch):
    """A pre-P0 SQLite DB (no agent_id/declared_* columns) gains them when opened via conn()."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db = tmp_path / "legacy.db"
    raw = sqlite3.connect(db)
    raw.executescript(
        "CREATE TABLE game_players (run_id TEXT, gid INTEGER, seat INTEGER, agent TEXT, model TEXT, "
        "  dealt_role TEXT, end_role TEXT, team TEXT, won INTEGER);"
        "CREATE TABLE agents (id TEXT PRIMARY KEY, display_name TEXT, token_hash TEXT UNIQUE, "
        "  protocol_version TEXT, sdk_version TEXT, created_utc TEXT, last_seen_utc TEXT, status TEXT);"
    )
    raw.commit()
    raw.close()
    monkeypatch.setattr(store, "DB_PATH", db)
    with store.conn() as c:  # opening runs _migrate (ALTER on open)
        gp = {r["name"] for r in c.execute("PRAGMA table_info(game_players)").fetchall()}
        ag = {r["name"] for r in c.execute("PRAGMA table_info(agents)").fetchall()}
    assert {"agent_id", "signup_id"} <= gp
    assert {"declared_model", "declared_harness"} <= ag


def test_register_agent_stores_declared_model_and_harness(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    a = store.register_agent("Vishal", "hash_v", "arena-agent-v1", "sdk-1",
                             declared_model="mistralai/mistral-small", declared_harness="base")
    assert a["declared_model"] == "mistralai/mistral-small"
    assert a["declared_harness"] == "base"
    listed = store.list_agents()
    assert any(x["id"] == a["id"] and x["declared_model"] == "mistralai/mistral-small" for x in listed)


def test_save_game_persists_agent_id_and_signup_id(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    meta = [
        {"name": "A", "model": "connected-agent", "harness": "connected",
         "agent_id": "agent_a", "signup_id": "sg_a"},
        {"name": "B", "model": "connected-agent", "harness": "connected",
         "agent_id": "agent_b", "signup_id": "sg_b"},
    ]
    assert store.save_game("run_x", 1, _transcript(), meta) is True
    with store.conn() as c:
        rows = {r["seat"]: dict(r) for r in c.execute(
            "SELECT seat, agent_id, signup_id FROM game_players WHERE run_id='run_x'").fetchall()}
    assert rows[0]["agent_id"] == "agent_a" and rows[0]["signup_id"] == "sg_a"
    assert rows[1]["agent_id"] == "agent_b" and rows[1]["signup_id"] == "sg_b"


def test_save_game_static_roster_leaves_agent_id_null(tmp_path, monkeypatch):
    """Static agents.yaml rosters have no agent_id; the column must be NULL, not an error."""
    _sqlite(tmp_path, monkeypatch)
    meta = [
        {"name": "Leo", "model": "phi-4", "harness": "base"},
        {"name": "Neeraj", "model": "gpt-4o-mini", "harness": "base"},
    ]
    assert store.save_game("run_static", 1, _transcript(), meta) is True
    with store.conn() as c:
        rows = c.execute(
            "SELECT agent_id FROM game_players WHERE run_id='run_static'").fetchall()
    assert all(r["agent_id"] is None for r in rows)


def test_backfill_connected_game_player_identities_matches_by_name_not_seat(tmp_path, monkeypatch):
    """Legacy connected rows predate agent_id columns; player rotation means seat is not a safe key."""
    _sqlite(tmp_path, monkeypatch)
    store.save_run({
        "id": "run_legacy", "game": "onuw", "label": "ONUW", "status": "done",
        "n_games": 1, "players": 2, "seed_base": 1, "created": "2026-01-01 00:00",
        "created_utc": "2026-01-01T00:00:00Z",
        "agents": [
            {"name": "Alice", "model": "connected-agent", "harness": "connected",
             "agent_id": "agent_alice", "signup_id": "signup_alice"},
            {"name": "Bob", "model": "connected-agent", "harness": "connected",
             "agent_id": "agent_bob", "signup_id": "signup_bob"},
        ],
    })
    transcript = {
        "seed": 1, "winner_team": "good", "outcome": {"text": "village wins"},
        "players": [
            {"seat": 0, "dealt": "Werewolf", "end": "Werewolf", "team": "evil",
             "won": False, "calls": 1, "forfeits": 0},
            {"seat": 1, "dealt": "Villager", "end": "Villager", "team": "good",
             "won": True, "calls": 1, "forfeits": 0},
        ],
    }
    assert store.save_game("run_legacy", 1, transcript, [
        {"name": "Bob", "model": "connected-agent", "harness": "connected"},
        {"name": "Alice", "model": "connected-agent", "harness": "connected"},
    ]) is True

    summary = store.backfill_connected_game_player_identities()
    assert summary["rows_updated"] == 2
    with store.conn() as c:
        rows = {r["seat"]: dict(r) for r in c.execute(
            "SELECT seat, agent, agent_id, signup_id FROM game_players WHERE run_id='run_legacy'"
        ).fetchall()}
    assert rows[0]["agent"] == "Bob"
    assert rows[0]["agent_id"] == "agent_bob"
    assert rows[0]["signup_id"] == "signup_bob"
    assert rows[1]["agent"] == "Alice"
    assert rows[1]["agent_id"] == "agent_alice"
    assert rows[1]["signup_id"] == "signup_alice"


# --- end-to-end: a real connected ONUW game writes agent_id-bearing rows -----------------------

def _active_signup(run_id, agent_name):
    agent = store.register_agent(agent_name, f"hash_{agent_name}", "arena-agent-v1", "test")
    signup, err = store.create_signup(run_id, agent["id"])
    assert err is None
    ready, err = store.mark_signup_ready(signup["id"], agent["id"])
    assert err is None or err == "not_ready_required"
    return agent["id"], ready["id"]


def _action_for(turn):
    kind = turn["action_kind"]
    players = turn["legal_action"].get("choices", {}).get("players") or []
    if kind == "onuw.discussion.speak_or_pass":
        return {"pass": True}
    if kind == "onuw.vote":
        return {"target": NO_ONE_REF}
    if kind == "onuw.seer.inspect":
        return {"mode": "center", "indices": [0, 1]}
    if kind == "onuw.troublemaker.swap_two_or_decline":
        return {"a": None, "b": None}
    if kind in {"onuw.doppelganger.copy_player", "onuw.robber.swap_or_decline"}:
        return {"target": players[0].get("ref", players[0].get("seat")) if players else None}
    if kind == "onuw.drunk.swap_center":
        return {"index": 0}
    raise AssertionError(kind)


def test_connected_game_writes_agent_id_rows(tmp_path, monkeypatch):
    from arena.connected import run_connected_batch
    _sqlite(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "run_link", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 1, "players": 5, "seed_base": 7,
    })
    agent_by_signup = {}
    for i in range(5):
        agent_id, signup_id = _active_signup("run_link", f"agent-{i}")
        agent_by_signup[signup_id] = agent_id
    for signup_id, agent_id in agent_by_signup.items():
        store.mark_signup_ready(signup_id, agent_id)
    assert {s["status"] for s in store.list_run_signups("run_link")} == {"active"}

    stop = threading.Event()

    def responder():
        while not stop.is_set():
            for signup_id, agent_id in agent_by_signup.items():
                turn = store.pending_turn_for_signup(signup_id)
                if turn:
                    store.reply_to_turn(turn["id"], agent_id, _action_for(turn), "scripted", 1)
            if {s["status"] for s in store.list_run_signups("run_link")} == {"completed"}:
                return
            time.sleep(0.01)

    thread = threading.Thread(target=responder)
    thread.start()
    try:
        run_connected_batch("run_link", discussion_rounds=1)
    finally:
        stop.set()
    thread.join(timeout=2)

    valid_ids = set(agent_by_signup.values())
    with store.conn() as c:
        rows = c.execute(
            "SELECT seat, agent, agent_id, signup_id FROM game_players WHERE run_id='run_link'").fetchall()
    assert len(rows) == 5  # one game, five seats
    for r in rows:
        assert r["agent_id"] in valid_ids, f"seat {r['seat']} missing/unknown agent_id: {dict(r)}"
        assert r["signup_id"] is not None
