"""Connected-agent coordinator primitives.

The game core still calls a synchronous Agent-like object. This adapter turns that call into a
durable turn row, waits for the SDK/API reply, and falls back deterministically at the deadline.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from . import store
from .batch import fresh_deal_schedule
from .games.onuw import ONUW
from .openrouter import AgentResponse


def _utc_deadline(value: str) -> _dt.datetime:
    return _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


@dataclass(frozen=True)
class ConnectedSeat:
    seat: int
    signup_id: str
    agent_id: str
    name: str


class ConnectedEventSink:
    def __init__(self, run_id: str, game_instance_id: str, seats: dict[int, ConnectedSeat]):
        self.run_id = run_id
        self.game_instance_id = game_instance_id
        self.seats = seats

    def __call__(self, event_type: str, payload: dict[str, Any], *,
                 phase: str | None = None, visibility: str = "public",
                 target_seat: int | None = None) -> None:
        target_signup_id = None
        if visibility == "private":
            if target_seat is None or target_seat not in self.seats:
                return
            target_signup_id = self.seats[target_seat].signup_id
        store.append_event(
            self.run_id,
            event_type,
            payload,
            visibility=visibility,
            target_signup_id=target_signup_id,
            game_instance_id=self.game_instance_id,
            phase=phase,
        )


@dataclass
class ConnectedTurnRequest:
    agent: "ConnectedAgent"
    turn: dict
    parse_action: Callable[[Any, str], Any]
    default_action: Any
    default_wire_action: Any
    started: float

    def wait(self) -> AgentResponse:
        turn_id = self.turn["id"]
        deadline = _utc_deadline(self.turn["deadline_utc"])
        last_raw = ""
        while _now() < deadline:
            reply = store.get_turn_reply(turn_id)
            if reply:
                return self._response_from_reply(reply)
            time.sleep(self.agent.poll_interval_s)

        reply = store.default_turn(turn_id, self.default_wire_action)
        if reply:
            parsed = self.default_action
            ms = (time.perf_counter() - self.started) * 1000
            raw = json.dumps(self.default_wire_action)
            self.agent.calls.append({"ok": False, "ms": ms, "raw": raw})
            store.append_event(
                self.agent.run_id,
                "forfeit",
                {"seat": self.agent.seat, "turn_id": turn_id, "reason": "deadline_expired"},
                game_instance_id=self.agent.game_instance_id,
                phase=self.turn["phase"],
            )
            return AgentResponse(
                reasoning="(deadline expired; defaulted)",
                action=parsed,
                raw=raw,
                ok=False,
                ms=ms,
            )
        return AgentResponse(
            reasoning="(turn disappeared; defaulted)",
            action=self.default_action,
            raw="",
            ok=False,
            ms=(time.perf_counter() - self.started) * 1000,
        )

    def _response_from_reply(self, reply: dict) -> AgentResponse:
        raw = json.dumps(reply["action"])
        ms = (time.perf_counter() - self.started) * 1000
        try:
            action = self.parse_action(reply["action"], raw)
            ok = bool(reply.get("accepted", 1))
        except (ValueError, KeyError, TypeError, AttributeError):
            action = self.default_action
            ok = False
        self.agent.calls.append({"ok": ok, "ms": ms, "raw": raw})
        store.append_event(
            self.agent.run_id,
            "action_result",
            {"seat": self.agent.seat, "turn_id": self.turn["id"], "accepted": ok},
            visibility="private",
            target_signup_id=self.agent.signup_id,
            game_instance_id=self.agent.game_instance_id,
            phase=self.turn["phase"],
        )
        return AgentResponse(
            reasoning=reply.get("reasoning") or "",
            action=action,
            raw=raw,
            ok=ok,
            ms=ms,
        )


class ConnectedAgent:
    model = "connected-agent"
    harness = "connected"

    def __init__(self, name: str, run_id: str, signup_id: str, seat: int,
                 game_instance_id: str, deadline_seconds: int = 60,
                 poll_interval_s: float = 0.1):
        self.name = name
        self.run_id = run_id
        self.signup_id = signup_id
        self.seat = seat
        self.game_instance_id = game_instance_id
        self.deadline_seconds = deadline_seconds
        self.poll_interval_s = poll_interval_s
        self.calls: list[dict] = []

    def prepare_act(
        self,
        observation: str,
        parse_action: Callable[[Any, str], Any],
        default_action: Any,
        *,
        action_kind: str,
        legal_action: dict,
        phase: str,
        default_wire_action: Any | None = None,
        deadline_seconds: int | None = None,
        **_: Any,
    ) -> ConnectedTurnRequest:
        turn = store.create_turn(
            self.run_id,
            self.signup_id,
            self.game_instance_id,
            self.seat,
            phase,
            action_kind,
            {"format": "text", "text": observation},
            legal_action,
            deadline_seconds=deadline_seconds or self.deadline_seconds,
        )
        store.append_event(
            self.run_id,
            "private_observation",
            {"seat": self.seat, "turn_id": turn["id"], "action_kind": action_kind},
            visibility="private",
            target_signup_id=self.signup_id,
            game_instance_id=self.game_instance_id,
            phase=phase,
        )
        return ConnectedTurnRequest(
            agent=self,
            turn=turn,
            parse_action=parse_action,
            default_action=default_action,
            default_wire_action=default_wire_action if default_wire_action is not None else default_action,
            started=time.perf_counter(),
        )

    def act(
        self,
        observation: str,
        parse_action: Callable[[Any, str], Any],
        default_action: Any,
        **turn_meta: Any,
    ) -> AgentResponse:
        return self.prepare_act(observation, parse_action, default_action, **turn_meta).wait()


def _connected_roster(run_id: str) -> list[ConnectedSeat]:
    signups = store.list_run_signups(run_id, statuses={"active"})
    seats = []
    for s in signups:
        if s.get("seat") is None:
            continue
        seats.append(ConnectedSeat(
            seat=int(s["seat"]),
            signup_id=s["id"],
            agent_id=s["agent_id"],
            name=s["display_name"],
        ))
    return sorted(seats, key=lambda s: s.seat)


def run_connected_batch(run_id: str, workers: int = 1, discussion_rounds: int | None = None) -> str:
    """Coordinate a connected run against the active store backend.

    This process may run locally while the API and agents talk to the same Neon database. Connected
    signups are intentionally serialized by default because v1 signups advertise one concurrent turn.
    """
    run = store.get_run(run_id)
    if not run:
        raise ValueError(f"run not found: {run_id}")
    if run["game"] != "onuw":
        raise ValueError("connected runner currently supports onuw")
    if workers != 1:
        raise ValueError("connected v1 requires workers=1 unless signups allow concurrent turns")

    seats = _connected_roster(run_id)
    if len(seats) != int(run["players"]):
        raise ValueError(f"run needs {run['players']} active signups, got {len(seats)}")
    n_games = int(run["n_games"])
    n_players = int(run["players"])
    rounds = discussion_rounds or 5
    failures: list[int] = []
    store.update_run_status(run_id, "running")

    for gid, seed, rot in fresh_deal_schedule(n_games, n_players, int(run["seed_base"])):
        game_instance_id = f"{run_id}_game_{gid:03d}"
        rotated = [seats[(i + rot) % n_players] for i in range(n_players)]
        seat_map = {
            i: ConnectedSeat(
                seat=i,
                signup_id=rotated[i].signup_id,
                agent_id=rotated[i].agent_id,
                name=rotated[i].name,
            )
            for i in range(n_players)
        }
        names = {i: seat_map[i].name for i in range(n_players)}
        agents = {
            i: ConnectedAgent(names[i], run_id, seat_map[i].signup_id, i, game_instance_id)
            for i in range(n_players)
        }
        store.append_event(run_id, "game_started", {"gid": gid, "seed": seed},
                           game_instance_id=game_instance_id, phase="setup")
        try:
            transcript = ONUW(
                names,
                seed=seed,
                discussion_rounds=rounds,
                event_sink=ConnectedEventSink(run_id, game_instance_id, seat_map),
                event_driven=True,   # delta transport: turns carry no context; harness rebuilds from events
            ).play(agents)
            for p in transcript["players"]:
                p["name"], p["model"] = names[p["seat"]], "connected-agent"
            meta = [
                {"name": names[i], "model": "connected-agent", "harness": "connected",
                 "agent_id": seat_map[i].agent_id, "signup_id": seat_map[i].signup_id}
                for i in range(n_players)
            ]
            store.save_game(run_id, gid, transcript, meta)
            store.append_event(
                run_id,
                "game_result",
                {"gid": gid, "winner_team": transcript["winner_team"], "text": transcript["outcome"]["text"]},
                game_instance_id=game_instance_id,
                phase="result",
            )
        except Exception as exc:
            failures.append(gid)
            store.append_event(run_id, "forfeit", {"gid": gid, "reason": f"{type(exc).__name__}: {exc}"},
                               game_instance_id=game_instance_id, phase="run")

    status = "done" if not failures else "partial"
    store.update_run_status(run_id, status)
    store.update_run_signups_status(run_id, "completed", {"active", "ready", "ready_required"})
    store.append_event(run_id, "run_completed", {"status": status, "failures": failures}, phase="run")
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a connected-agent Persuasion Arena run")
    parser.add_argument("--run", required=True, help="run_id to coordinate")
    parser.add_argument("--rounds", type=int, default=5, help="discussion rounds")
    args = parser.parse_args(argv)
    run_connected_batch(args.run, discussion_rounds=args.rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
