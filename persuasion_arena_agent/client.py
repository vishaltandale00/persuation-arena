from __future__ import annotations

import email.utils
import time
from typing import Any

import httpx

from .credentials import AgentCredentials, DEFAULT_SERVER
from .models import PollResponse, Signup


class ArenaHttpClient:
    def __init__(self, server: str = DEFAULT_SERVER, transport: httpx.BaseTransport | None = None,
                 timeout: float = 30.0, max_retries: int = 3):
        self.server = server.rstrip("/")
        self._client = httpx.Client(base_url=self.server, transport=transport, timeout=timeout)
        self.max_retries = max_retries

    def close(self) -> None:
        self._client.close()

    def _retry_after_s(self, response: httpx.Response | None) -> float | None:
        if not response:
            return None
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            if parsed.tzinfo is None:
                return None
            return max(0.0, parsed.timestamp() - time.time())

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        transient_statuses = {408, 425, 429, 500, 502, 503, 504}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(method, path, **kwargs)
                if response.status_code in transient_statuses and attempt < self.max_retries:
                    retry_after = self._retry_after_s(response)
                    delay = retry_after if retry_after is not None else min(0.25 * (2 ** attempt), 2.0)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response
            except httpx.TransportError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(0.25 * (2 ** attempt), 2.0))
        assert last_error is not None
        raise last_error

    def register_agent(self, display_name: str, protocol_version: str = "arena-agent-v1",
                       sdk_version: str = "0.1.0") -> AgentCredentials:
        r = self._request("POST", "/api/agents/register", json={
            "display_name": display_name,
            "protocol_version": protocol_version,
            "sdk_version": sdk_version,
        })
        d = r.json()
        return AgentCredentials(
            server=self.server,
            agent_id=d["agent_id"],
            display_name=display_name,
            agent_token=d["agent_token"],
        )

    def discover_runs(self, game: str | None = None) -> list[dict[str, Any]]:
        params = {"game": game} if game else None
        r = self._request("GET", "/api/runs/open", params=params)
        return r.json().get("runs", [])

    def signup_run(self, creds: AgentCredentials, run_id: str,
                   max_concurrent_turns: int = 1) -> Signup:
        r = self._request(
            "POST",
            f"/api/runs/{run_id}/signups",
            headers=creds.auth_header(),
            json={"protocol_version": "arena-agent-v1", "max_concurrent_turns": max_concurrent_turns},
        )
        return Signup.from_dict(r.json())

    def get_signup(self, creds: AgentCredentials, signup_id: str) -> Signup:
        r = self._request("GET", f"/api/signups/{signup_id}", headers=creds.auth_header())
        return Signup.from_dict(r.json())

    def mark_ready(self, creds: AgentCredentials, signup_id: str) -> Signup:
        r = self._request(
            "POST",
            f"/api/signups/{signup_id}/ready",
            headers=creds.auth_header(),
            json={"protocol_version": "arena-agent-v1", "sdk_version": "0.1.0"},
        )
        return Signup.from_dict(r.json())

    def poll_signup(self, creds: AgentCredentials, signup_id: str, after_event_id: str | None = None,
                    max_events: int = 50) -> PollResponse:
        r = self._request(
            "POST",
            f"/api/signups/{signup_id}/poll",
            headers=creds.auth_header(),
            json={"after_event_id": after_event_id, "max_events": max_events},
        )
        return PollResponse.from_dict(r.json())

    def reply_turn(self, creds: AgentCredentials, turn_id: str, action: Any,
                   reasoning: str | None = None, client_ms: int | None = None) -> dict:
        r = self._request(
            "POST",
            f"/api/turns/{turn_id}/reply",
            headers=creds.auth_header(),
            json={"action": action, "reasoning": reasoning, "client_ms": client_ms},
        )
        return r.json()
