"""Deterministic state probe agent for testing memory scope.

This agent has no LLM. It records every delivered event under the same state key policy as the
stateful reference harnesses, then includes that key and event count in discussion messages and
private reasoning. Use it to smoke-test ARENA_AGENT_RESET_BETWEEN_GAMES against a real connected run.

Optional:
    ARENA_STATE_PROBE_LOG=/tmp/probe.jsonl
"""
from __future__ import annotations

import json
import os
import time

from examples._harness_util import render_event, reset_between_games_from_env, state_key


class StateProbeAgent:
    def __init__(self, reset_between_games: bool | None = None, log_path: str | None = None):
        self.reset_between_games = (reset_between_games if reset_between_games is not None
                                    else reset_between_games_from_env())
        self.log_path = log_path or os.environ.get("ARENA_STATE_PROBE_LOG")
        self.events: dict[str, list[str]] = {}
        self.turns: dict[str, int] = {}

    def _log(self, kind: str, key: str, obj) -> None:
        if not self.log_path:
            return
        row = {
            "ts": time.time(),
            "kind": kind,
            "key": key,
            "run_id": getattr(obj, "run_id", None),
            "game_instance_id": getattr(obj, "game_instance_id", None),
            "reset_between_games": self.reset_between_games,
            "events_seen": len(self.events.get(key, [])),
            "turns_seen": self.turns.get(key, 0),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def on_event(self, event) -> None:
        key = state_key(event, self.reset_between_games)
        if key is None:
            return
        self.events.setdefault(key, []).append(render_event(event))
        self._log("event", key, event)

    def act(self, turn) -> dict:
        key = state_key(turn, self.reset_between_games) or turn.game_instance_id
        self.turns[key] = self.turns.get(key, 0) + 1
        events_seen = len(self.events.get(key, []))
        reasoning = (
            f"state_key={key}; reset_between_games={self.reset_between_games}; "
            f"events_seen={events_seen}; turns_seen={self.turns[key]}"
        )
        self._log("turn", key, turn)

        kind = turn.action_kind
        players = [p["seat"] for p in (turn.legal_action.get("choices", {}).get("players") or [])]
        if kind == "onuw.discussion.speak_or_pass":
            return {"action": {"speak": f"probe {reasoning}"}, "reasoning": reasoning}
        if kind == "onuw.vote":
            return {"action": {"target": -1}, "reasoning": reasoning}
        if kind == "onuw.seer.inspect":
            return {"action": {"mode": "center", "indices": [0, 1]}, "reasoning": reasoning}
        if kind == "onuw.troublemaker.swap_two_or_decline":
            return {"action": {"a": None, "b": None}, "reasoning": reasoning}
        if kind in {"onuw.doppelganger.copy_player", "onuw.robber.swap_or_decline"}:
            return {"action": {"target": players[0] if players else None}, "reasoning": reasoning}
        if kind == "onuw.drunk.swap_center":
            return {"action": {"index": 0}, "reasoning": reasoning}
        return {"action": {"pass": True}, "reasoning": reasoning}


_default = StateProbeAgent()
on_event = _default.on_event
act = _default.act
