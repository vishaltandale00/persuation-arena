"""OpenRouter-backed agent.

An agent receives a filtered text observation + a JSON action schema for the turn, and returns
a parsed {"declared_reasoning": str, "action": ...}. That field is private agent-authored rationale
logged by the arena. OpenRouter/provider-native reasoning is requested and stored separately as
`provider_reasoning` when the provider returns it.

On a malformed/failed response we retry once, then fall back to a safe default the caller supplies.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable

from openai import OpenAI

from ._jsonparse import extract_json
from .config import Caps, OPENROUTER_BASE_URL, SETTINGS, get_api_key
from .wolf_profiles import prompt_for

_client: OpenAI | None = None


def openrouter_client(api_key: str | None = None) -> OpenAI:
    """The single OpenRouter-backed OpenAI client. Shared by this module and the example
    harnesses so the base URL + key resolution live in exactly one place. `api_key` lets a
    caller (e.g. a harness reading OPENROUTER_API_KEY directly) supply its own key; when omitted
    we fall back to the arena config's resolver."""
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key or get_api_key())
    return _client


def client() -> OpenAI:
    return openrouter_client()


def reasoning_extra_body(effort: str, *, compression: bool = False) -> dict[str, Any]:
    """The `extra_body` we send OpenRouter: reasoning config, and optionally the
    context-compression plugin. Compression is an EXPLICIT opt-in so callers that unify on this
    helper do not silently change their request shape."""
    body: dict[str, Any] = {
        "reasoning": {
            "effort": effort,
            "exclude": False,
        },
    }
    if compression:
        body["plugins"] = [
            {"id": "context-compression", "enabled": True},
        ]
    return body


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

_COMPOSITION_SCHEMA_KEYWORDS = {"oneOf", "anyOf"}
_UNSUPPORTED_JSON_SCHEMA_KEYWORDS = {
    "allOf",
    "not",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "then",
    "else",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minimum",
    "maximum",
    "multipleOf",
    "patternProperties",
    "unevaluatedProperties",
    "propertyNames",
    "minProperties",
    "maxProperties",
    "unevaluatedItems",
    "contains",
    "minContains",
    "maxContains",
    "minItems",
    "maxItems",
    "uniqueItems",
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
        validation_error: str | None = None,
    ) -> None:
        self.declared_reasoning = declared_reasoning
        self.action = action
        self.raw = raw
        self.ok = ok
        self.ms = ms
        self.provider_reasoning = provider_reasoning
        self.provider_reasoning_details = provider_reasoning_details
        self.validation_error = validation_error


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
        "required": ["declared_reasoning", "action"],
        "properties": {
            "declared_reasoning": {"type": "string"},
            "action": action_schema or {},
        },
        "additionalProperties": False,
    }


def _type_list(schema: dict[str, Any]) -> list[str]:
    typ = schema.get("type")
    if isinstance(typ, list):
        return [str(t) for t in typ]
    if isinstance(typ, str):
        return [typ]
    if "enum" in schema:
        types = []
        for value in schema["enum"]:
            if value is None:
                t = "null"
            elif isinstance(value, bool):
                t = "boolean"
            elif isinstance(value, int):
                t = "integer"
            elif isinstance(value, float):
                t = "number"
            elif isinstance(value, str):
                t = "string"
            else:
                t = "object"
            if t not in types:
                types.append(t)
        return types
    return []


def _ordered_union(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _schema_type_value(types: list[str]) -> str | list[str]:
    ordered = [t for t in ["string", "integer", "number", "boolean", "object", "array"] if t in types]
    if not ordered:
        return "string"
    if ordered == ["integer", "number"]:
        return "number"
    return ordered[0]


def _sentinel_for_type(typ: str) -> Any:
    return {
        "string": "",
        "integer": 0,
        "number": 0,
        "boolean": False,
        "array": [],
        "object": {},
    }.get(typ, "")


def _merge_property_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    types: list[str] = []
    enums: list[Any] = []
    has_non_null_enum = False
    array_items: list[dict[str, Any]] = []
    object_props: dict[str, Any] = {}
    for schema in schemas:
        safe = _provider_safe_json_schema(schema)
        if safe is None:
            safe = {"type": "null"}
        for typ in _type_list(safe):
            if typ not in types:
                types.append(typ)
        if "enum" in safe:
            enums.extend(safe["enum"])
            has_non_null_enum = has_non_null_enum or any(value is not None for value in safe["enum"])
        if safe.get("type") == "array" or (isinstance(safe.get("type"), list) and "array" in safe["type"]):
            items = safe.get("items")
            if isinstance(items, dict):
                array_items.append(items)
        if safe.get("type") == "object" and isinstance(safe.get("properties"), dict):
            object_props.update(safe["properties"])

    non_null_types = [typ for typ in types if typ != "null"]
    merged_type = _schema_type_value(non_null_types)
    merged: dict[str, Any] = {"type": merged_type}
    if enums and has_non_null_enum:
        normalized_enums = [
            _sentinel_for_type(str(merged_type)) if value is None else value
            for value in enums
        ]
        sentinel = _sentinel_for_type(str(merged_type))
        if "null" in types and sentinel not in normalized_enums:
            normalized_enums.insert(0, sentinel)
        merged["enum"] = _ordered_union(normalized_enums)
    if "array" in non_null_types:
        merged["items"] = _merge_property_schemas(array_items) if array_items else {}
    if "object" in non_null_types and object_props:
        merged["properties"] = object_props
        merged["required"] = list(object_props)
        merged["additionalProperties"] = False
    return merged


def _flatten_composed_object_schema(schema: dict[str, Any], keyword: str) -> dict[str, Any] | None:
    branches = schema.get(keyword)
    if not isinstance(branches, list) or not branches:
        return None
    safe_branches: list[dict[str, Any]] = []
    for branch in branches:
        safe = _provider_safe_json_schema(branch)
        if not isinstance(safe, dict) or safe.get("type") != "object" or not isinstance(safe.get("properties"), dict):
            return None
        safe_branches.append(safe)

    prop_names: list[str] = []
    for branch in safe_branches:
        for name in branch["properties"]:
            if name not in prop_names:
                prop_names.append(name)

    properties: dict[str, Any] = {}
    for name in prop_names:
        variants: list[dict[str, Any]] = []
        for branch in safe_branches:
            if name in branch["properties"] and name in set(branch.get("required") or []):
                variants.append(branch["properties"][name])
            else:
                variants.append({"type": "null"})
        properties[name] = _merge_property_schemas(variants)

    return {
        "type": "object",
        "required": prop_names,
        "properties": properties,
        "additionalProperties": False,
    }


def _provider_safe_json_schema(schema: Any) -> dict[str, Any] | None:
    if not isinstance(schema, dict):
        return None
    for keyword in _COMPOSITION_SCHEMA_KEYWORDS:
        if keyword in schema:
            return _flatten_composed_object_schema(schema, keyword)

    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_JSON_SCHEMA_KEYWORDS:
            continue
        if key == "type" and isinstance(value, list):
            non_null_types = [str(item) for item in value if item != "null"]
            out[key] = _schema_type_value(non_null_types)
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {
                name: _provider_safe_json_schema(prop_schema) or {}
                for name, prop_schema in value.items()
            }
        elif key == "items":
            out[key] = _provider_safe_json_schema(value) or {}
        elif isinstance(value, dict):
            safe = _provider_safe_json_schema(value)
            if safe is not None:
                out[key] = safe
        elif isinstance(value, list):
            out[key] = value
        else:
            out[key] = value

    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    if isinstance(out.get("type"), str) and isinstance(out.get("enum"), list):
        sentinel = _sentinel_for_type(out["type"])
        out["enum"] = _ordered_union([
            sentinel if value is None else value
            for value in out["enum"]
        ])
    return out


def _has_unsupported_json_schema_keywords(schema: Any) -> bool:
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in _COMPOSITION_SCHEMA_KEYWORDS or key in _UNSUPPORTED_JSON_SCHEMA_KEYWORDS:
                return True
            if _has_unsupported_json_schema_keywords(value):
                return True
    elif isinstance(schema, list):
        return any(_has_unsupported_json_schema_keywords(item) for item in schema)
    return False


def _looks_like_structured_rejection(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "response_format",
        "json_schema",
        "json_object",
        "output_config.format",
        "text.format.schema",
        "schema at ",
        "invalid schema",
        "invalid_argument",
        "requires unspecified property",
        "schema type",
        "oneof",
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
        if caps.prior_message_turns < 0:
            prior = self._messages
        else:
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
        validation_error: str | None = None,
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

    def _request_extra_body(
        self,
        *,
        reasoning_effort: str | None = None,
        exclude_reasoning: bool | None = None,
    ) -> dict[str, Any]:
        # The arena client opts INTO context-compression; harnesses keep it off (see
        # reasoning_extra_body). Keeping the opt-in explicit means unifying the helper does not
        # silently change either caller's request shape.
        body = reasoning_extra_body(reasoning_effort or self.caps.reasoning_effort, compression=True)
        if exclude_reasoning is not None:
            body["reasoning"]["exclude"] = bool(exclude_reasoning)
        return body

    def _response_format(self, action_kind: str | None, action_schema: dict[str, Any] | None) -> dict[str, Any] | None:
        mode = getattr(self.caps, "openrouter_structured_output", "off")
        if mode == "off":
            return None
        if mode == "auto":
            if _model_provider(self.model) not in _STRUCTURED_SCHEMA_PROVIDERS:
                return None
            mode = "json_schema" if action_schema else "json_object"
        if mode == "json_schema" and action_schema:
            safe_action_schema = _provider_safe_json_schema(action_schema)
            if safe_action_schema is None or _has_unsupported_json_schema_keywords(safe_action_schema):
                return {"type": "json_object"}
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": _structured_schema_name(action_kind),
                    "strict": True,
                    "schema": _response_schema(safe_action_schema),
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
        suppress_reasoning: bool = False,
    ) -> Any:
        caps = self.caps
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": caps.max_tokens_per_turn,
            "timeout": caps.request_timeout_s,
            "extra_body": self._request_extra_body(
                reasoning_effort="none" if suppress_reasoning else None,
                exclude_reasoning=True if suppress_reasoning else None,
            ),
        }
        if caps.temperature is not None:
            kwargs["temperature"] = caps.temperature
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
        suppress_reasoning_retry = False
        suppress_reasoning_retry_used = False
        messages = self._messages_for_turn(observation)
        t0 = time.perf_counter()
        for attempt in range(caps.retries + 2):
            if attempt > caps.retries and not suppress_reasoning_retry:
                break
            suppress_reasoning = suppress_reasoning_retry
            suppress_reasoning_retry = False
            try:
                resp = self._create_completion(
                    messages=messages,
                    response_format=response_format,
                    suppress_reasoning=suppress_reasoning,
                )
            except Exception as e:  # network/timeout/api error
                if response_format is not None and _looks_like_structured_rejection(e):
                    last_validation_error = f"structured output rejected ({type(e).__name__}: {e})"
                    structured_fallback = True
                    response_format = None
                    try:
                        resp = self._create_completion(
                            messages=messages,
                            response_format=None,
                            suppress_reasoning=suppress_reasoning,
                        )
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

            obj = extract_json(last_raw)
            if obj is None:
                last_validation_error = "response did not contain a JSON object"
                if not suppress_reasoning_retry_used and (provider_reasoning is not None or not last_raw):
                    suppress_reasoning_retry_used = True
                    suppress_reasoning_retry = True
                continue
            declared_reasoning = str(obj.get("declared_reasoning", obj.get("reasoning", ""))).strip()
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
                "suppress_reasoning_retry": suppress_reasoning,
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
                validation_error=last_validation_error,
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
            "suppress_reasoning_retry": suppress_reasoning_retry_used,
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
            validation_error=last_validation_error or last_error,
        )
