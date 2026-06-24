"""Most-basic stateful harness: keep one LLM chat session alive per game by default.

Set ARENA_AGENT_RESET_BETWEEN_GAMES=0 to keep one chat session alive for the whole run instead.

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

from examples._harness_util import (
    SYSTEM, DEFAULT_MODEL, render_event, action_request, decide,
    reset_between_games_from_env, state_key,
)


class SessionAgent:
    def __init__(self, model: str = DEFAULT_MODEL, reset_between_games: bool | None = None):
        self.model = model
        self.reset_between_games = (reset_between_games if reset_between_games is not None
                                    else reset_between_games_from_env())
        self.sessions: dict[str, list[dict]] = {}   # state key -> live chat messages

    def _session(self, key: str) -> list[dict]:
        s = self.sessions.get(key)
        if s is None:
            s = [{"role": "system", "content": SYSTEM}]
            self.sessions[key] = s
        return s

    def on_event(self, event) -> None:
        key = state_key(event, self.reset_between_games)
        if key is None:
            return
        self._session(key).append({"role": "user", "content": render_event(event)})

    def act(self, turn) -> dict:
        key = state_key(turn, self.reset_between_games) or turn.game_instance_id
        s = self._session(key)
        s.append({"role": "user", "content": action_request(turn)})
        action, reasoning, assistant = decide(self.model, s, turn)
        s.append({"role": "assistant", "content": assistant})  # keep our own reasoning as memory
        return {"action": action, "reasoning": reasoning}


# Module-level handlers for `arena-agent play ... examples/session_agent.py` (model via env).
_default = SessionAgent(os.environ.get("ARENA_AGENT_MODEL", DEFAULT_MODEL))
on_event = _default.on_event
act = _default.act
