"""Deterministic agents for tests — no API calls."""
from __future__ import annotations


class _Resp:
    def __init__(self, reasoning, action):
        self.declared_reasoning = reasoning
        self.action = action
        self.ok = True
        self.raw = ""
        self.ms = 0.0


class ScriptedDefault:
    """Always takes the engine's safe default action."""
    name = "S"
    model = "scripted"

    def act(self, observation, parse_action, default_action):
        return _Resp("(default)", default_action)


class PolicyAgent:
    """Returns a raw action when an observation matches a policy key, else the default.

    policy: {substring_in_prompt: raw_action}. Raw actions are validated via parse_action.
    record: optional list; vote-phase observations are appended for atomicity checks.
    """
    name = "P"
    model = "scripted"

    def __init__(self, policy=None, record=None):
        self.policy = policy or {}
        self.record = record

    def act(self, observation, parse_action, default_action):
        if self.record is not None and "FINAL VOTE" in observation:
            self.record.append(observation)
        raw = None
        for key, val in self.policy.items():
            if key in observation:
                raw = val
                break
        if raw is None:
            action = default_action
        else:
            try:
                action = parse_action(raw, "")  # validate; fall back like the real agent does
            except (ValueError, KeyError, TypeError):
                action = default_action
        return _Resp("(policy)", action)
