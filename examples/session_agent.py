"""Most-basic stateful harness: keep one LLM chat session alive per game.

Memory = the conversation. `on_event` appends each delta event as a message; `act` appends the
action request, calls the LLM with the whole session, and keeps the LLM's own reply in the session
so its prior reasoning persists. The harness never re-fetches past events (it can't — the server
delivers each once and never re-serves them); its memory is whatever it kept in the session.

This is deliberately minimal — a reference for building a harness. A smarter harness might compress
its memory instead of keeping the full conversation; that's the participant's choice.

Wire it onto an ArenaAgent (the SDK delivers events + turns; the harness owns memory):
    sa = SessionAgent("openai/gpt-4o-mini")
    agent.on_event(sa.on_event)
    agent.act(sa.act)
"""
from __future__ import annotations

import os

from examples._harness_util import SYSTEM, DEFAULT_MODEL, render_event, action_request, decide


class SessionAgent:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.sessions: dict[str, list[dict]] = {}   # game_instance_id -> live chat messages

    def _session(self, gid: str) -> list[dict]:
        s = self.sessions.get(gid)
        if s is None:
            s = [{"role": "system", "content": SYSTEM}]
            self.sessions[gid] = s
        return s

    def on_event(self, event) -> None:
        gid = getattr(event, "game_instance_id", None)
        if gid is None:
            return  # run-level event — not part of any game's memory
        self._session(gid).append({"role": "user", "content": render_event(event)})

    def act(self, turn) -> dict:
        s = self._session(turn.game_instance_id)
        s.append({"role": "user", "content": action_request(turn)})
        action, reasoning, assistant = decide(self.model, s, turn)
        s.append(assistant)  # keep provider-native reasoning fields when OpenRouter returns them
        return {"action": action, "reasoning": reasoning}


# Module-level handlers for `arena-agent play ... examples/session_agent.py` (model via env).
_default = SessionAgent(os.environ.get("ARENA_AGENT_MODEL", DEFAULT_MODEL))
on_event = _default.on_event
act = _default.act
