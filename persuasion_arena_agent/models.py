from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Signup:
    signup_id: str
    run_id: str
    agent_id: str
    status: str
    seat: int | None = None
    poll_after_ms: int = 1000
    heartbeat_after_ms: int | None = None
    waiting_expires_at: str | None = None
    ready_deadline_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Signup":
        return cls(
            signup_id=data["signup_id"],
            run_id=data["run_id"],
            agent_id=data["agent_id"],
            status=data["status"],
            seat=data.get("seat"),
            poll_after_ms=int(data.get("poll_after_ms") or 1000),
            heartbeat_after_ms=data.get("heartbeat_after_ms"),
            waiting_expires_at=data.get("waiting_expires_at"),
            ready_deadline_at=data.get("ready_deadline_at"),
        )


@dataclass(frozen=True)
class Event:
    event_id: str
    type: str
    payload: dict[str, Any]
    seq: int | None = None
    game_instance_id: str | None = None
    visibility: str | None = None
    phase: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        return cls(
            event_id=data["event_id"],
            type=data["type"],
            payload=data.get("payload") or {},
            seq=data.get("seq"),
            game_instance_id=data.get("game_instance_id"),
            visibility=data.get("visibility"),
            phase=data.get("phase"),
        )


@dataclass(frozen=True)
class Turn:
    turn_id: str
    game_instance_id: str
    game: str
    seat: int
    phase: str
    action_kind: str
    deadline_at: str
    observation: dict[str, Any]
    legal_action: dict[str, Any]

    @property
    def legal_actions(self) -> dict[str, Any]:
        return self.legal_action

    @classmethod
    def from_dict(cls, data: dict) -> "Turn":
        return cls(
            turn_id=data["turn_id"],
            game_instance_id=data["game_instance_id"],
            game=data["game"],
            seat=int(data["seat"]),
            phase=data["phase"],
            action_kind=data["action_kind"],
            deadline_at=data["deadline_at"],
            observation=data.get("observation") or {},
            legal_action=data.get("legal_action") or {},
        )


@dataclass(frozen=True)
class PollResponse:
    signup_id: str
    run_id: str
    run_status: str
    events: list[Event]
    turn: Turn | None
    poll_after_ms: int

    @classmethod
    def from_dict(cls, data: dict) -> "PollResponse":
        return cls(
            signup_id=data["signup_id"],
            run_id=data["run_id"],
            run_status=data["run_status"],
            events=[Event.from_dict(e) for e in data.get("events", [])],
            turn=Turn.from_dict(data["turn"]) if data.get("turn") else None,
            poll_after_ms=int(data.get("poll_after_ms") or 1000),
        )
