from __future__ import annotations

import json
import os

import httpx

from persuasion_arena_agent import ArenaAgent, AgentCredentials, CredentialsStore, redact_token
from persuasion_arena_agent.client import ArenaHttpClient


def test_credentials_store_reuses_and_redacts_token(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    monkeypatch.setenv("PERSUASION_ARENA_CREDENTIALS", str(path))
    store = CredentialsStore()
    creds = AgentCredentials(
        server="https://example.test",
        agent_id="agent_1",
        display_name="agent",
        agent_token="pa_live_super_secret_token",
    )
    store.save(creds)

    loaded = store.get("https://example.test")
    assert loaded == creds
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert "super_secret" not in repr(loaded)
    assert redact_token(creds.agent_token) != creds.agent_token


def test_http_client_uses_protocol_paths_and_bearer_headers():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.headers.get("authorization"),
                      json.loads(request.content or b"{}")))
        if request.url.path == "/api/agents/register":
            return httpx.Response(200, json={"agent_id": "agent_1", "agent_token": "pa_live_token",
                                             "protocol_version": "arena-agent-v1"})
        if request.url.path == "/api/runs/open":
            return httpx.Response(200, json={"runs": [{"run_id": "run_1"}]})
        if request.url.path == "/api/runs/run_1/signups":
            return httpx.Response(200, json={"signup_id": "signup_1", "run_id": "run_1",
                                             "agent_id": "agent_1", "status": "waiting"})
        if request.url.path == "/api/signups/signup_1":
            return httpx.Response(200, json={"signup_id": "signup_1", "run_id": "run_1",
                                             "agent_id": "agent_1", "status": "ready_required"})
        if request.url.path == "/api/signups/signup_1/ready":
            return httpx.Response(200, json={"signup_id": "signup_1", "run_id": "run_1",
                                             "agent_id": "agent_1", "status": "active"})
        if request.url.path == "/api/signups/signup_1/poll":
            return httpx.Response(200, json={"signup_id": "signup_1", "run_id": "run_1",
                                             "run_status": "active", "events": [],
                                             "turn": None, "poll_after_ms": 250})
        if request.url.path == "/api/turns/turn_1/reply":
            return httpx.Response(200, json={"ok": True, "accepted": True})
        return httpx.Response(404, json={"path": request.url.path})

    client = ArenaHttpClient("https://example.test", transport=httpx.MockTransport(handler))
    creds = client.register_agent("agent", model="m-x", harness="h-y")
    assert creds.agent_token == "pa_live_token"
    assert client.discover_runs("onuw")[0]["run_id"] == "run_1"
    signup = client.signup_run(creds, "run_1")
    client.get_signup(creds, signup.signup_id)
    client.mark_ready(creds, signup.signup_id)
    client.poll_signup(creds, signup.signup_id)
    client.reply_turn(creds, "turn_1", {"pass": True}, "ok", 1)

    assert (calls[0][0], calls[0][1]) == ("POST", "/api/agents/register")
    assert calls[0][3]["model"] == "m-x"
    assert calls[0][3]["harness"] == "h-y"
    assert any(c[1] == "/api/runs/run_1/signups" and c[2] == "Bearer pa_live_token" for c in calls)
    assert any(c[1] == "/api/signups/signup_1/poll" for c in calls)


def test_http_client_retries_transient_status_with_retry_after(monkeypatch):
    calls = {"n": 0}
    sleeps = []

    monkeypatch.setattr("persuasion_arena_agent.client.time.sleep", lambda delay: sleeps.append(delay))

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"}, json={"detail": "busy"})
        return httpx.Response(200, json={"runs": [{"run_id": "run_retry"}]})

    client = ArenaHttpClient("https://example.test", transport=httpx.MockTransport(handler),
                             max_retries=2)

    assert client.discover_runs("onuw") == [{"run_id": "run_retry"}]
    assert calls["n"] == 2
    assert sleeps == [0.25]


def test_arena_agent_ready_poll_act_and_event_cursor(tmp_path):
    path = tmp_path / "credentials.json"
    calls = {"polls": [], "acts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        if request.url.path == "/api/agents/register":
            return httpx.Response(200, json={"agent_id": "agent_1", "agent_token": "pa_live_token",
                                             "protocol_version": "arena-agent-v1"})
        if request.url.path == "/api/runs/run_1/signups":
            return httpx.Response(200, json={"signup_id": "signup_1", "run_id": "run_1",
                                             "agent_id": "agent_1", "status": "ready_required"})
        if request.url.path == "/api/signups/signup_1" and request.method == "GET":
            return httpx.Response(200, json={"signup_id": "signup_1", "run_id": "run_1",
                                             "agent_id": "agent_1", "status": "ready_required"})
        if request.url.path == "/api/signups/signup_1/ready":
            return httpx.Response(200, json={"signup_id": "signup_1", "run_id": "run_1",
                                             "agent_id": "agent_1", "status": "active"})
        if request.url.path == "/api/signups/signup_1/poll":
            calls["polls"].append(body)
            turn = None if len(calls["polls"]) > 1 else {
                "turn_id": "turn_1",
                "game_instance_id": "game_1",
                "game": "onuw",
                "seat": 0,
                "phase": "discussion",
                "action_kind": "onuw.discussion.speak_or_pass",
                "deadline_at": "2026-06-23T17:00:00Z",
                "observation": {"format": "text", "text": "speak"},
                "legal_action": {"schema": {}, "choices": {}},
            }
            return httpx.Response(200, json={
                "signup_id": "signup_1", "run_id": "run_1", "run_status": "active",
                "events": [{"event_id": f"evt_{len(calls['polls'])}", "type": "speech",
                            "payload": {}, "seq": len(calls["polls"])}],
                "turn": turn,
                "poll_after_ms": 250,
            })
        if request.url.path == "/api/turns/turn_1/reply":
            return httpx.Response(200, json={"ok": True, "accepted": True})
        return httpx.Response(404)

    client = ArenaHttpClient("https://example.test", transport=httpx.MockTransport(handler))
    agent = ArenaAgent("agent", "https://example.test", CredentialsStore(path), client)

    @agent.act
    def act(turn):
        calls["acts"] += 1
        assert turn.observation["text"] == "speak"
        return {"action": {"speak": "hello"}, "reasoning": "test"}

    signup = agent.signup(run_id="run_1")
    agent.run_once([signup])
    agent.run_once([signup])

    assert calls["acts"] == 1
    assert calls["polls"][0]["after_event_id"] is None
    assert calls["polls"][1]["after_event_id"] == "evt_1"
    assert os.environ.get("PERSUASION_ARENA_CREDENTIALS") is None
