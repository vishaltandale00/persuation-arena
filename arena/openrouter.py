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

from .config import Caps, OPENROUTER_BASE_URL, SETTINGS, get_api_key
from .wolf_profiles import prompt_for

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=get_api_key())
    return _client


# Compatibility alias for code or tests that import SYSTEM directly.
SYSTEM = prompt_for("base")

_STRUCTURED_SCHEMA_PROVIDERS = {
    "anthropic",
    "google",
    "mistralai",
    "openai",
    "x-ai",
    "z-ai",
}


class AgentResponse:
    def __init__(
        self,
        *,
        action: Any,
        raw: str,
        ok: bool,
        ms: float = 0.0,
        declared_reasoning: str = "",
        provider_reasoning: Any = None,
        provider_reasoning_details: Any = None,
    ) -> None:
        self.declared_reasoning = declared_reasoning
        self.action = action
        self.raw = raw
        self.ok = ok
        self.ms = ms
        self.provider_reasoning = provider_reasoning
        self.provider_reasoning_details = provider_reasoning_details


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


def _model_provider(model: str) -> str:
    return model.split("/", 1)[0].strip().lower()


def _action_schema_from_turn_meta(turn_meta: dict[str, Any]) -> dict[str, Any] | None:
    action_schema = turn_meta.get("action_schema")
    if isinstance(action_schema, dict):
        return action_schema
    legal_action = turn_meta.get("legal_action")
    if isinstance(legal_action, dict) and isinstance(legal_action.get("schema"), dict):
        return legal_action["schema"]
    return None


def _structured_schema_name(action_kind: str | None) -> str:
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", action_kind or "arena_action").strip("_")
    if not base:
        base = "arena_action"
    if not re.match(r"^[a-zA-Z]", base):
        base = f"arena_{base}"
    return base[:64]


def _response_schema(action_schema: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["reasoning", "action"],
        "properties": {
            "reasoning": {"type": "string"},
            "action": action_schema or {},
        },
        "additionalProperties": False,
    }


def _looks_like_structured_rejection(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "response_format",
        "json_schema",
        "json_object",
        "structured output",
        "structured outputs",
        "unsupported parameter",
        "unexpected keyword argument",
    )
    return any(needle in text for needle in needles)


class OpenRouterAgent:
    def __init__(self, name: str, model: str, harness: str = "base", caps: Caps | None = None):
        self.name = name
        self.model = model
        self.harness = harness
        self.caps = caps or SETTINGS.caps
        # Per-turn telemetry. Existing score code only depends on {ok, ms, raw}; the richer fields
        # are for audit/debug and are safe for old leaderboard rows to ignore.
        self.calls: list[dict] = []
        self._messages: list[dict[str, Any]] = []

    def _messages_for_turn(self, observation: str) -> list[dict[str, Any]]:
        caps = self.caps
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
        caps = self.caps
        return {
            "reasoning": {
                "effort": caps.reasoning_effort,
                "exclude": False,
            },
        }

    def _response_format(self, action_kind: str | None, action_schema: dict[str, Any] | None) -> dict[str, Any] | None:
        mode = getattr(SETTINGS.caps, "openrouter_structured_output", "off")
        if mode == "off":
            return None
        if mode == "auto":
            if _model_provider(self.model) not in _STRUCTURED_SCHEMA_PROVIDERS:
                return None
            mode = "json_schema" if action_schema else "json_object"
        if mode == "json_schema" and action_schema:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": _structured_schema_name(action_kind),
                    "strict": True,
                    "schema": _response_schema(action_schema),
                },
            }
        if mode in {"json_object", "json_schema"}:
            return {"type": "json_object"}
        return None

    def _create_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None,
    ) -> Any:
        caps = SETTINGS.caps
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": caps.max_tokens_per_turn,
            "temperature": caps.temperature,
            "timeout": caps.request_timeout_s,
            "extra_body": self._request_extra_body(),
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        return client().chat.completions.create(**kwargs)

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
        caps = self.caps
        last_raw = ""
        last_validation_error: str | None = None
        last_error: str | None = None
        provider_reasoning = None
        provider_reasoning_details = None
        finish_reason = None
        usage: dict[str, Any] = {}
        action_kind = _.get("action_kind")
        action_schema = _action_schema_from_turn_meta(_)
        response_format = self._response_format(action_kind, action_schema)
        requested_response_format_type = (
            response_format.get("type") if isinstance(response_format, dict) else None
        )
        structured_output_config = getattr(caps, "openrouter_structured_output", "off")
        structured_fallback = False
        messages = self._messages_for_turn(observation)
        t0 = time.perf_counter()
        for attempt in range(caps.retries + 1):
            try:
                resp = self._create_completion(messages=messages, response_format=response_format)
            except Exception as e:  # network/timeout/api error
                if response_format is not None and _looks_like_structured_rejection(e):
                    last_validation_error = f"structured output rejected ({type(e).__name__}: {e})"
                    structured_fallback = True
                    response_format = None
                    try:
                        resp = self._create_completion(messages=messages, response_format=None)
                    except Exception as fallback_e:
                        last_raw = f"<error: {type(fallback_e).__name__}>"
                        last_error = f"{type(fallback_e).__name__}: {fallback_e}"
                        continue
                else:
                    last_raw = f"<error: {type(e).__name__}>"
                    last_error = f"{type(e).__name__}: {e}"
                    continue

            try:
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
            except Exception as e:
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
                "temperature": caps.temperature,
                "retries": caps.retries,
                "prior_message_turns": caps.prior_message_turns,
                "finish_reason": finish_reason,
                "usage": usage,
                "validation_error": last_validation_error,
                "structured_output": {
                    "configured": structured_output_config,
                    "requested": requested_response_format_type,
                    "used": response_format.get("type") if isinstance(response_format, dict) else None,
                    "fallback": structured_fallback,
                },
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
            "temperature": caps.temperature,
            "retries": caps.retries,
            "prior_message_turns": caps.prior_message_turns,
            "finish_reason": finish_reason,
            "usage": usage,
            "validation_error": last_validation_error or last_error,
            "structured_output": {
                "configured": structured_output_config,
                "requested": requested_response_format_type,
                "used": response_format.get("type") if isinstance(response_format, dict) else None,
                "fallback": structured_fallback,
            },
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
