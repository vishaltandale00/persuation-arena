"""VDD: explicit seat (roster index) threads end-to-end through the signup wire (FINDING P2).

The orchestrator's deterministic-seat schedule (SPEC D5/V-7) is only honored if every shard host can
request an EXPLICIT seat at signup time. store.create_signup(..., seat=) already supports it, but the
SDK signup path (agent.signup -> client.signup_run -> POST /signups) only threaded join_token, never the
seat. So children seated by arrival order per shard, breaking the V-7 fairness guarantee in production.

These drive the REAL SDK -> server (TestClient) -> store path. seat=k must land in the store; omitting it
must keep arrival-order seating byte-identical to before (INV-2).
"""
from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from arena import store
from persuasion_arena_agent import ArenaAgent, CredentialsStore
from persuasion_arena_agent.client import ArenaHttpClient


def _server_client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("INGEST_TOKENS", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "seat_wire.db")
    from arena.server import app
    return TestClient(app)


def _sdk_agent(name: str, server: str, transport: httpx.BaseTransport, cred_path) -> ArenaAgent:
    client = ArenaHttpClient(server, transport=transport)
    return ArenaAgent(name=name, server=server, credentials=CredentialsStore(cred_path), client=client)


def test_signup_request_body_includes_seat_only_when_set():
    """client.signup_run must include `seat` in the body iff it is set (omit for normal runs, INV-2)."""
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        bodies.append(_json.loads(request.content or b"{}"))
        return httpx.Response(200, json={"signup_id": "s1", "run_id": "run_1",
                                         "agent_id": "a1", "status": "waiting"})

    client = ArenaHttpClient("https://example.test", transport=httpx.MockTransport(handler))
    from persuasion_arena_agent.credentials import AgentCredentials
    creds = AgentCredentials(server="https://example.test", agent_id="a1",
                             display_name="n", agent_token="pa_live_x")

    client.signup_run(creds, "run_1")
    assert "seat" not in bodies[-1], "normal signup must NOT carry a seat (INV-2)"

    client.signup_run(creds, "run_1", seat=3)
    assert bodies[-1].get("seat") == 3, "explicit seat must be wired into the request body"


def test_agent_signup_threads_seat(tmp_path):
    captured = {}

    class _StubClient(ArenaHttpClient):
        def __init__(self):
            pass

        def signup_run(self, creds, run_id, max_concurrent_turns=1, join_token=None, seat=None):
            captured["seat"] = seat
            from persuasion_arena_agent.models import Signup
            return Signup.from_dict({"signup_id": "s1", "run_id": run_id,
                                     "agent_id": "a1", "status": "waiting"})

    agent = ArenaAgent(name="n", server="https://example.test",
                       credentials=CredentialsStore(tmp_path / "c.json"), client=_StubClient())
    agent.credentials.save  # noqa: B018
    from persuasion_arena_agent.credentials import AgentCredentials
    agent.credentials.save(AgentCredentials(server=agent.server, agent_id="a1",
                                            display_name="n", agent_token="pa_live_x"))
    agent.signup(run_id="run_1", seat=2)
    assert captured["seat"] == 2, "agent.signup must thread seat to client.signup_run"


def test_explicit_seat_lands_in_store_via_real_path(tmp_path, monkeypatch):
    """End-to-end: SDK -> /api/runs/{id}/signups -> store. seat=k => store seats at k after ready."""
    with _server_client(tmp_path, monkeypatch) as server:
        created = server.post("/api/runs", json={
            "connected": True, "run_id": "run_seat", "game": "onuw",
            "players": 5, "games": 1, "seed": 7001,
        })
        assert created.status_code == 200, created.text

        # Drive through the SDK client but pointed at the in-process TestClient transport.
        sdk_transport = httpx.MockTransport(lambda req: _proxy(server, req))
        names = ["alpha", "bravo", "charlie", "delta", "echo"]
        # Roster index we WANT each agent to land in (reverse of arrival order to prove it's honored).
        wanted = {"alpha": 4, "bravo": 3, "charlie": 2, "delta": 1, "echo": 0}
        signup_ids = {}
        for name in names:
            agent = _sdk_agent(name, "http://server.test", sdk_transport, tmp_path / f"{name}.json")
            sg = agent.signup(run_id="run_seat", seat=wanted[name])
            signup_ids[name] = sg.signup_id

        for name in names:
            r = server.post(f"/api/signups/{signup_ids[name]}/ready",
                            json={"protocol_version": "arena-agent-v1"},
                            headers=_auth_for(server, name))
            assert r.status_code == 200, r.text

        for name, want in wanted.items():
            st = server.get(f"/api/signups/{signup_ids[name]}", headers=_auth_for(server, name))
            assert st.status_code == 200, st.text
            assert st.json()["seat"] == want, f"{name} expected seat {want}, got {st.json()['seat']}"


def test_no_seat_keeps_arrival_order(tmp_path, monkeypatch):
    """INV-2: signups WITHOUT an explicit seat seat in arrival order, byte-identical to before."""
    with _server_client(tmp_path, monkeypatch) as server:
        created = server.post("/api/runs", json={
            "connected": True, "run_id": "run_arr", "game": "onuw",
            "players": 5, "games": 1, "seed": 7002,
        })
        assert created.status_code == 200, created.text
        sdk_transport = httpx.MockTransport(lambda req: _proxy(server, req))
        order = ["xa", "xb", "xc", "xd", "xe"]
        signup_ids = {}
        for name in order:
            agent = _sdk_agent(name, "http://server.test", sdk_transport, tmp_path / f"{name}.json")
            sg = agent.signup(run_id="run_arr")  # no seat
            signup_ids[name] = sg.signup_id
        for name in order:
            r = server.post(f"/api/signups/{signup_ids[name]}/ready",
                            json={"protocol_version": "arena-agent-v1"},
                            headers=_auth_for(server, name))
            assert r.status_code == 200, r.text
        for seat, name in enumerate(order):
            st = server.get(f"/api/signups/{signup_ids[name]}", headers=_auth_for(server, name))
            assert st.json()["seat"] == seat, f"{name} expected arrival seat {seat}"


# --- helpers to bridge the SDK httpx client onto the in-process TestClient ----------------------
_TOKENS: dict[str, str] = {}


def _auth_for(server: TestClient, name: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {_TOKENS[name]}"}


def _proxy(server: TestClient, req: httpx.Request) -> httpx.Response:
    """Forward an SDK httpx request into the in-process FastAPI TestClient and capture register tokens."""
    import json as _json
    body = _json.loads(req.content or b"{}") if req.content else None
    headers = {k: v for k, v in req.headers.items() if k.lower() == "authorization"}
    path = req.url.path
    resp = server.request(req.method, path, json=body, headers=headers)
    if path == "/api/agents/register" and resp.status_code == 200:
        _TOKENS[body["display_name"]] = resp.json()["agent_token"]
    return httpx.Response(resp.status_code, json=resp.json() if resp.content else {})
