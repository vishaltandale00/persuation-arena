from __future__ import annotations

from fastapi.testclient import TestClient

from arena import store


def _client(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("INGEST_TOKENS", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "agent_protocol.db")
    from arena.server import app
    return TestClient(app)


def _register(client: TestClient, name: str):
    r = client.post("/api/agents/register", json={
        "display_name": name,
        "protocol_version": "arena-agent-v1",
        "sdk_version": "test",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    return d["agent_id"], d["agent_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_register_stores_only_token_hash(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        agent_id, token = _register(client, "sharp-wolf")
        row = store.get_agent(agent_id)
        assert row["display_name"] == "sharp-wolf"
        assert row["token_hash"] != token
        assert token not in row["token_hash"]
        assert token.startswith("pa_live_")


def test_signup_duplicate_ready_gate_and_open_run_filter(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        created = client.post("/api/runs", json={
            "connected": True, "run_id": "run_ready", "game": "onuw",
            "players": 5, "games": 1, "seed": 9000,
        })
        assert created.status_code == 200, created.text
        tokens = [_register(client, f"agent-{i}")[1] for i in range(5)]

        first = client.post("/api/runs/run_ready/signups", headers=_auth(tokens[0]),
                            json={"protocol_version": "arena-agent-v1"})
        assert first.status_code == 200, first.text
        first_id = first.json()["signup_id"]
        assert first.json()["status"] == "waiting"

        duplicate = client.post("/api/runs/run_ready/signups", headers=_auth(tokens[0]),
                                json={"protocol_version": "arena-agent-v1"})
        assert duplicate.status_code == 200, duplicate.text
        assert duplicate.json()["signup_id"] == first_id

        signup_ids = [first_id]
        for token in tokens[1:]:
            r = client.post("/api/runs/run_ready/signups", headers=_auth(token),
                            json={"protocol_version": "arena-agent-v1"})
            assert r.status_code == 200, r.text
            signup_ids.append(r.json()["signup_id"])

        st = client.get(f"/api/signups/{first_id}", headers=_auth(tokens[0]))
        assert st.status_code == 200, st.text
        assert st.json()["status"] == "ready_required"
        assert st.json()["seat"] == 0

        open_runs = client.get("/api/runs/open?game=onuw")
        assert open_runs.status_code == 200
        assert all(r["run_id"] != "run_ready" for r in open_runs.json()["runs"])

        for token, signup_id in zip(tokens[:-1], signup_ids[:-1], strict=True):
            r = client.post(f"/api/signups/{signup_id}/ready", headers=_auth(token),
                            json={"protocol_version": "arena-agent-v1"})
            assert r.status_code == 200, r.text
            assert r.json()["status"] == "ready"

        last = client.post(f"/api/signups/{signup_ids[-1]}/ready", headers=_auth(tokens[-1]),
                           json={"protocol_version": "arena-agent-v1"})
        assert last.status_code == 200, last.text
        assert last.json()["status"] == "active"

        st = client.get(f"/api/signups/{first_id}", headers=_auth(tokens[0]))
        assert st.json()["status"] == "active"


def test_agent_auth_scopes_signup_access(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        store.create_connected_run({
            "id": "run_auth", "game": "onuw", "label": "ONUW", "status": "open",
            "n_games": 1, "players": 1, "seed_base": 1,
        })
        _, token_a = _register(client, "a")
        _, token_b = _register(client, "b")
        signup = client.post("/api/runs/run_auth/signups", headers=_auth(token_a),
                             json={"protocol_version": "arena-agent-v1"}).json()

        no_token = client.get(f"/api/signups/{signup['signup_id']}")
        assert no_token.status_code == 401

        wrong = client.get(f"/api/signups/{signup['signup_id']}", headers=_auth(token_b))
        assert wrong.status_code == 404


def test_poll_returns_events_turn_and_first_reply_wins(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        store.create_connected_run({
            "id": "run_stream", "game": "onuw", "label": "ONUW", "status": "open",
            "n_games": 1, "players": 1, "seed_base": 2,
        })
        _, token = _register(client, "streamer")
        signup = client.post("/api/runs/run_stream/signups", headers=_auth(token),
                             json={"protocol_version": "arena-agent-v1"}).json()
        signup_id = signup["signup_id"]
        ready = client.post(f"/api/signups/{signup_id}/ready", headers=_auth(token),
                            json={"protocol_version": "arena-agent-v1"})
        assert ready.status_code == 200, ready.text
        assert ready.json()["status"] == "active"

        public = store.append_event("run_stream", "speech", {"actor_seat": 0, "text": "hello"},
                                    phase="discussion")
        private = store.append_event("run_stream", "role_info", {"role": "Seer"},
                                     visibility="private", target_signup_id=signup_id, phase="night")
        store.append_event("run_stream", "role_info", {"role": "Werewolf"},
                           visibility="private", target_signup_id="signup_other", phase="night")
        turn = store.create_turn(
            "run_stream", signup_id, "game_001", 0, "discussion",
            "onuw.discussion.speak_or_pass",
            {"format": "text", "text": "speak now"},
            {"schema": {"type": "object"}, "choices": {}},
        )

        poll = client.post(f"/api/signups/{signup_id}/poll", headers=_auth(token),
                           json={"after_event_id": None})
        assert poll.status_code == 200, poll.text
        body = poll.json()
        visible_action_events = [e for e in body["events"] if e["type"] in {"speech", "role_info"}]
        assert [e["event_id"] for e in visible_action_events] == [public["id"], private["id"]]
        assert body["turn"]["turn_id"] == turn["id"]
        assert body["turn"]["action_kind"] == "onuw.discussion.speak_or_pass"
        assert body["turn"]["observation"]["text"] == "speak now"

        later = store.append_event("run_stream", "pass", {"actor_seat": 0}, phase="discussion")
        poll2 = client.post(f"/api/signups/{signup_id}/poll", headers=_auth(token),
                            json={"after_event_id": private["id"]})
        assert [e["event_id"] for e in poll2.json()["events"]] == [later["id"]]

        first = client.post(f"/api/turns/{turn['id']}/reply", headers=_auth(token),
                            json={"action": {"speak": "first"}, "reasoning": "r1", "client_ms": 1})
        assert first.status_code == 200, first.text
        assert first.json() == {"ok": True, "accepted": True}

        second = client.post(f"/api/turns/{turn['id']}/reply", headers=_auth(token),
                             json={"action": {"speak": "second"}, "reasoning": "r2", "client_ms": 2})
        assert second.status_code == 200, second.text
        assert second.json()["accepted"] is False

        with store.conn() as c:
            row = c.execute("SELECT action_json FROM turn_replies WHERE turn_id=?", (turn["id"],)).fetchone()
        assert row["action_json"] == '{"speak": "first"}'


def test_run_debug_exposes_connected_observer_state(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        store.create_connected_run({
            "id": "run_debug", "game": "onuw", "label": "ONUW", "status": "open",
            "n_games": 1, "players": 1, "seed_base": 3,
        })
        _, token = _register(client, "debugger")
        signup = client.post("/api/runs/run_debug/signups", headers=_auth(token),
                             json={"protocol_version": "arena-agent-v1"}).json()
        signup_id = signup["signup_id"]
        client.post(f"/api/signups/{signup_id}/ready", headers=_auth(token),
                    json={"protocol_version": "arena-agent-v1"})
        event = store.append_event("run_debug", "speech", {"actor_seat": 0, "text": "hello"},
                                   phase="discussion")
        turn = store.create_turn(
            "run_debug", signup_id, "game_1", 0, "discussion",
            "onuw.discussion.speak_or_pass",
            {"format": "text", "text": "speak"},
            {"schema": {"type": "object", "required": ["speak"],
                        "properties": {"speak": {"type": "string"}},
                        "additionalProperties": False}},
        )

        overview = client.get("/api/runs/run_debug")
        assert overview.status_code == 200
        agent = overview.json()["agents"][0]
        assert agent["signup_id"] == signup_id
        assert agent["status"] == "active"

        debug = client.get("/api/runs/run_debug/debug")
        assert debug.status_code == 200, debug.text
        body = debug.json()
        assert body["connected"] is True
        assert body["signups"][0]["signup_id"] == signup_id
        assert body["pending_turns"][0]["turn_id"] == turn["id"]
        assert event["id"] in [e["event_id"] for e in body["recent_events"]]
