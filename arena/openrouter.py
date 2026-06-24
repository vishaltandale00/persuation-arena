"""OpenRouter-backed agent.

An agent receives a filtered text observation + a JSON action schema for the turn, and returns
a parsed {"reasoning": str, "action": ...}. That JSON field is `declared_reasoning`: private
agent-authored rationale logged by the arena. OpenRouter/provider-native reasoning is requested and
stored separately as `provider_reasoning` when the provider returns it.

On a malformed/failed response we retry once, then fall back to a safe default the caller supplies.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from openai import OpenAI

from .config import OPENROUTER_BASE_URL, SETTINGS, get_api_key
from .wolf_profiles import prompt_for

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=get_api_key())
    return _client


# Compatibility alias for code or tests that import SYSTEM directly.
SYSTEM = prompt_for("base")


class AgentResponse:
    def __init__(
        self,
        *,
        action: Any,
        raw: str,
        ok: bool,
        ms: float = 0.0,
        declared_reasoning: str | None = None,
        # Temporary compatibility only: migrate callers to explicit declared_reasoning and remove
        # this alias. Cleanup note:
        # /Users/jacobajit/.codex/memories/extensions/ad_hoc/notes/20260624T201527Z-persuation-arena-reasoning-compat-cleanup.md
        reasoning: str | None = None,
        provider_reasoning: Any = None,
        provider_reasoning_details: Any = None,
    ) -> None:
        self.declared_reasoning = declared_reasoning if declared_reasoning is not None else (reasoning or "")
        self.action = action
        self.raw = raw
        self.ok = ok
        self.ms = ms
        self.provider_reasoning = provider_reasoning
        self.provider_reasoning_details = provider_reasoning_details

    @property
    def reasoning(self) -> str:
        """Temporary alias for game cores/tests that still read `resp.reasoning`."""
        return self.declared_reasoning


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
        # Per-turn telemetry. Existing score code only depends on {ok, ms, raw}; the richer fields
        # are for audit/debug and are safe for old leaderboard rows to ignore.
        self.calls: list[dict] = []
        self._messages: list[dict[str, Any]] = []

    def _messages_for_turn(self, observation: str) -> list[dict[str, Any]]:
        caps = SETTINGS.caps
        prior = self._messages[-max(0, caps.prior_message_turns) * 2:]
        return [
            {"role": "system", "content": prompt_for(self.harness)},
            *prior,
            {"role": "user", "content": observation},
        ]

    def _remember_turn(
        self,
        *,
        observation: str,
        raw: str,
        provider_reasoning: Any,
        provider_reasoning_details: Any,
    ) -> None:
        assistant_message: dict[str, Any] = {"role": "assistant", "content": raw}
        if provider_reasoning_details is not None:
            assistant_message["reasoning_details"] = provider_reasoning_details
        elif provider_reasoning is not None:
            assistant_message["reasoning"] = provider_reasoning
        self._messages.extend([
            {"role": "user", "content": observation},
            assistant_message,
        ])

    def _request_extra_body(self) -> dict[str, Any]:
        caps = SETTINGS.caps
        return {
            "reasoning": {
                "effort": caps.reasoning_effort,
                "exclude": False,
            },
        }

    @staticmethod
    def _message_field(message: Any, field: str) -> Any:
        if isinstance(message, dict):
            return message.get(field)
        return getattr(message, field, None)

    def act(
        self,
        observation: str,
        parse_action: Callable[[Any, str], Any],
        default_action: Any,
        **_: Any,
    ) -> AgentResponse:
        """observation: full filtered prompt for this turn.
        parse_action(action_field, raw_text) -> validated action (raise ValueError if invalid).
        default_action: used if the model fails twice.
        """
        caps = SETTINGS.caps
        last_raw = ""
        last_validation_error: str | None = None
        last_error: str | None = None
        provider_reasoning = None
        provider_reasoning_details = None
        finish_reason = None
        usage: dict[str, Any] = {}
        action_kind = _.get("action_kind")
        messages = self._messages_for_turn(observation)
        t0 = time.perf_counter()
        for attempt in range(caps.retries + 1):
            try:
                resp = client().chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=caps.max_tokens_per_turn,
                    temperature=caps.temperature,
                    timeout=caps.request_timeout_s,
                    extra_body=self._request_extra_body(),
                )
                choice = resp.choices[0]
                message = choice.message
                last_raw = (self._message_field(message, "content") or "").strip()
                provider_reasoning = self._message_field(message, "reasoning")
                provider_reasoning_details = self._message_field(message, "reasoning_details")
                finish_reason = getattr(choice, "finish_reason", None)
                usage_obj = getattr(resp, "usage", None)
                usage = (
                    usage_obj.model_dump()
                    if hasattr(usage_obj, "model_dump")
                    else dict(usage_obj) if isinstance(usage_obj, dict) else {}
                )
            except Exception as e:  # network/timeout/api error
                last_raw = f"<error: {type(e).__name__}>"
                last_error = f"{type(e).__name__}: {e}"
                continue

            obj = _extract_json(last_raw)
            if obj is None:
                last_validation_error = "response did not contain a JSON object"
                continue
            declared_reasoning = str(obj.get("reasoning", "")).strip()
            try:
                action = parse_action(obj.get("action"), last_raw)
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                last_validation_error = f"{type(e).__name__}: {e}"
                continue
            ms = (time.perf_counter() - t0) * 1000
            record = {
                "ok": True,
                "ms": ms,
                "raw": last_raw,
                "action_kind": action_kind,
                "declared_reasoning": declared_reasoning,
                "provider_reasoning": provider_reasoning,
                "provider_reasoning_details": provider_reasoning_details,
                "reasoning_effort": caps.reasoning_effort,
                "max_tokens": caps.max_tokens_per_turn,
                "finish_reason": finish_reason,
                "usage": usage,
                "validation_error": last_validation_error,
            }
            self.calls.append(record)
            self._remember_turn(
                observation=observation,
                raw=last_raw,
                provider_reasoning=provider_reasoning,
                provider_reasoning_details=provider_reasoning_details,
            )
            return AgentResponse(
                declared_reasoning=declared_reasoning,
                action=action,
                raw=last_raw,
                ok=True,
                ms=ms,
                provider_reasoning=provider_reasoning,
                provider_reasoning_details=provider_reasoning_details,
            )

        ms = (time.perf_counter() - t0) * 1000
        # forfeit: keep the raw model output (if any) so the defaulted turn is auditable.
        declared_reasoning = "(no valid response - defaulted)"
        record = {
            "ok": False,
            "ms": ms,
            "raw": last_raw,
            "action_kind": action_kind,
            "declared_reasoning": declared_reasoning,
            "provider_reasoning": provider_reasoning,
            "provider_reasoning_details": provider_reasoning_details,
            "reasoning_effort": caps.reasoning_effort,
            "max_tokens": caps.max_tokens_per_turn,
            "finish_reason": finish_reason,
            "usage": usage,
            "validation_error": last_validation_error or last_error,
        }
        self.calls.append(record)
        return AgentResponse(
            declared_reasoning=declared_reasoning,
            action=default_action,
            raw=last_raw,
            ok=False,
            ms=ms,
            provider_reasoning=provider_reasoning,
            provider_reasoning_details=provider_reasoning_details,
        )
