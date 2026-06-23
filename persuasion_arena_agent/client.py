from __future__ import annotations

from typing import Any

import httpx

from .credentials import AgentCredentials, DEFAULT_SERVER
from .models import PollResponse, Signup


class ArenaHttpClient:
    def __init__(self, server: str = DEFAULT_SERVER, transport: httpx.BaseTransport | None = None,
                 timeout: float = 30.0):
        self.server = server.rstrip("/")
        self._client = httpx.Client(base_url=self.server, transport=transport, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def register_agent(self, display_name: str, protocol_version: str = "arena-agent-v1",
                       sdk_version: str = "0.1.0") -> AgentCredentials:
        r = self._client.post("/api/agents/register", json={
            "display_name": display_name,
            "protocol_version": protocol_version,
            "sdk_version": sdk_version,
        })
        r.raise_for_status()
        d = r.json()
        return AgentCredentials(
            server=self.server,
            agent_id=d["agent_id"],
            display_name=display_name,
            agent_token=d["agent_token"],
        )

    def discover_runs(self, game: str | None = None) -> list[dict[str, Any]]:
        params = {"game": game} if game else None
        r = self._client.get("/api/runs/open", params=params)
        r.raise_for_status()
        return r.json().get("runs", [])

    def signup_run(self, creds: AgentCredentials, run_id: str,
                   max_concurrent_turns: int = 1) -> Signup:
        r = self._client.post(
            f"/api/runs/{run_id}/signups",
            headers=creds.auth_header(),
            json={"protocol_version": "arena-agent-v1", "max_concurrent_turns": max_concurrent_turns},
        )
        r.raise_for_status()
        return Signup.from_dict(r.json())

    def get_signup(self, creds: AgentCredentials, signup_id: str) -> Signup:
        r = self._client.get(f"/api/signups/{signup_id}", headers=creds.auth_header())
        r.raise_for_status()
        return Signup.from_dict(r.json())

    def mark_ready(self, creds: AgentCredentials, signup_id: str) -> Signup:
        r = self._client.post(
            f"/api/signups/{signup_id}/ready",
            headers=creds.auth_header(),
            json={"protocol_version": "arena-agent-v1", "sdk_version": "0.1.0"},
        )
        r.raise_for_status()
        return Signup.from_dict(r.json())

    def poll_signup(self, creds: AgentCredentials, signup_id: str, after_event_id: str | None = None,
                    max_events: int = 50) -> PollResponse:
        r = self._client.post(
            f"/api/signups/{signup_id}/poll",
            headers=creds.auth_header(),
            json={"after_event_id": after_event_id, "max_events": max_events},
        )
        r.raise_for_status()
        return PollResponse.from_dict(r.json())

    def reply_turn(self, creds: AgentCredentials, turn_id: str, action: Any,
                   reasoning: str | None = None, client_ms: int | None = None) -> dict:
        r = self._client.post(
            f"/api/turns/{turn_id}/reply",
            headers=creds.auth_header(),
            json={"action": action, "reasoning": reasoning, "client_ms": client_ms},
        )
        r.raise_for_status()
        return r.json()
