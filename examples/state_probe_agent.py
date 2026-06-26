"""Deterministic state-probe harness for testing memory scope (no LLM, no network).

This harness records every delivered event and turn under the SAME state-key policy as the stateful
reference harnesses (`state_key(obj, reset_between_games)`), then reports that key plus the
event/turn counts it has accumulated under it. Use it to assert reset-per-game vs carry-across-run
behavior of ARENA_AGENT_RESET_BETWEEN_GAMES — in a unit test or against a real connected run.

Wire it onto an ArenaAgent like any other harness:
    p = StateProbeAgent()
    agent.on_event(p.on_event)
    agent.act(p.act)

Optional:
    ARENA_STATE_PROBE_LOG=/tmp/probe.jsonl   # append one JSON row per event/turn for inspection
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
        self.events: dict[str, list[str]] = {}   # state key -> rendered events seen under it
        self.turns: dict[str, int] = {}          # state key -> turns acted under it

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
            return  # run-level event with no game/run-scoped state key
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
        players = [p.get("ref", p.get("seat"))
                   for p in (turn.legal_action.get("choices", {}).get("players") or [])]
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


# Module-level handlers for `arena-agent play ... examples/state_probe_agent.py`.
_default = StateProbeAgent()
on_event = _default.on_event
act = _default.act
