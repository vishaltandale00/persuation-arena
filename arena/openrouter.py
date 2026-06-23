"""OpenRouter-backed agent.

An agent receives a filtered text observation + a JSON action schema for the turn, and returns
a parsed {"reasoning": str, "action": ...}. `reasoning` is private (logged, never shown to others).
On a malformed/failed response we retry once, then fall back to a safe default the caller supplies.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI

from .config import OPENROUTER_BASE_URL, SETTINGS, get_api_key

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=get_api_key())
    return _client


SYSTEM = (
    "You are a sharp, competitive player of a hidden-role social-deduction game. "
    "Read the situation, reason about who is lying and what serves your team, then act. "
    "Always reply with a single JSON object and nothing else: "
    '{"reasoning": "<your private thinking, never shown to others>", "action": <the action>}. '
    "Keep reasoning to a few sentences. Follow the action format the prompt specifies exactly."
)


@dataclass
class AgentResponse:
    reasoning: str
    action: Any
    raw: str
    ok: bool
    ms: float = 0.0   # wall-clock the agent took to decide (model latency, incl. retries)


def _extract_json(text: str) -> dict | None:
    # tolerate code fences / surrounding prose: grab the first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # try trimming trailing junk
        frag = m.group(0)
        for end in range(len(frag), 0, -1):
            try:
                return json.loads(frag[:end])
            except json.JSONDecodeError:
                continue
    return None


class OpenRouterAgent:
    def __init__(self, name: str, model: str, harness: str = "base"):
        self.name = name
        self.model = model
        self.harness = harness
        # per-turn telemetry: one record {ok, ms, raw} per act() call, in turn order.
        # ok=False marks a forfeited turn (the engine substituted a default action).
        self.calls: list[dict] = []

    def act(
        self,
        observation: str,
        parse_action: Callable[[Any, str], Any],
        default_action: Any,
    ) -> AgentResponse:
        """observation: full filtered prompt for this turn.
        parse_action(action_field, raw_text) -> validated action (raise ValueError if invalid).
        default_action: used if the model fails twice.
        """
        caps = SETTINGS.caps
        last_raw = ""
        t0 = time.perf_counter()
        for attempt in range(caps.retries + 1):
            try:
                resp = client().chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": observation},
                    ],
                    max_tokens=caps.max_tokens_per_turn,
                    temperature=caps.temperature,
                    timeout=caps.request_timeout_s,
                )
                last_raw = (resp.choices[0].message.content or "").strip()
            except Exception as e:  # network/timeout/api error
                last_raw = f"<error: {type(e).__name__}>"
                continue

            obj = _extract_json(last_raw)
            if obj is None:
                continue
            reasoning = str(obj.get("reasoning", "")).strip()
            try:
                action = parse_action(obj.get("action"), last_raw)
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
            ms = (time.perf_counter() - t0) * 1000
            self.calls.append({"ok": True, "ms": ms, "raw": last_raw})
            return AgentResponse(reasoning=reasoning, action=action, raw=last_raw, ok=True, ms=ms)

        ms = (time.perf_counter() - t0) * 1000
        # forfeit: keep the raw model output (if any) so the defaulted turn is auditable.
        self.calls.append({"ok": False, "ms": ms, "raw": last_raw})
        return AgentResponse(
            reasoning="(no valid response — defaulted)",
            action=default_action,
            raw=last_raw,
            ok=False,
            ms=ms,
        )
