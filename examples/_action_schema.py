"""Canonical client-side action validator for ONUW harnesses.

A connected harness must validate its action against the EXACT `turn.legal_action` payload before it
replies, or it can submit something the server rejects with 422 (e.g. an over-long discussion speech,
a Seer center-inspect with the wrong number of indices, or an unexpected field). This module is the
single client-side validator used by BOTH the generic reference harnesses (examples/_harness_util.py)
and WolfForgeV2 (examples/wolfforge_v2_policy.py), so neither relies on a looser hand-coded
approximation.

It is a faithful, model-free reimplementation of the small JSON-schema subset the server enforces in
`arena.store._validate_action` (oneOf/anyOf/enum/type/required/additionalProperties/properties, array
minItems/maxItems/uniqueItems/items, numeric minimum/maximum, string minLength/maxLength) plus the
`legal_action["rules"]["distinct"]` field-distinctness rule. We deliberately do NOT import the private
store function into participant code; instead a parity test (tests/test_harness_validation.py) pins
this validator against `arena.store._validate_action` over a battery of cases so any drift is caught.
"""
from __future__ import annotations

import json
from typing import Any


def _type_ok(value: Any, expected: Any) -> bool:
    kinds = expected if isinstance(expected, list) else [expected]
    for kind in kinds:
        if kind == "null" and value is None:
            return True
        if kind == "boolean" and isinstance(value, bool):
            return True
        if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if kind == "string" and isinstance(value, str):
            return True
        if kind == "array" and isinstance(value, list):
            return True
        if kind == "object" and isinstance(value, dict):
            return True
    return False


def _validate_json_schema(value: Any, schema: dict | None) -> tuple[bool, str | None]:
    if not schema:
        return True, None
    if "oneOf" in schema:
        errors = []
        for option in schema["oneOf"]:
            ok, err = _validate_json_schema(value, option)
            if ok:
                return True, None
            errors.append(err or "invalid")
        return False, "; ".join(errors[:2])
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            ok, _ = _validate_json_schema(value, option)
            if ok:
                return True, None
        return False, "does not match any allowed shape"
    if "enum" in schema and value not in schema["enum"]:
        return False, "not in enum"
    if "type" in schema and not _type_ok(value, schema["type"]):
        return False, f"expected {schema['type']}"

    if isinstance(value, dict):
        for key in schema.get("required") or []:
            if key not in value:
                return False, f"missing required field {key}"
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                return False, f"unexpected field {sorted(extra)[0]}"
        for key, subschema in properties.items():
            if key in value:
                ok, err = _validate_json_schema(value[key], subschema)
                if not ok:
                    return False, f"{key}: {err}"

    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            return False, "too few items"
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            return False, "too many items"
        if schema.get("uniqueItems") and len(value) != len({json.dumps(v, sort_keys=True) for v in value}):
            return False, "items must be unique"
        if "items" in schema:
            for item in value:
                ok, err = _validate_json_schema(item, schema["items"])
                if not ok:
                    return False, f"item: {err}"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False, "below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return False, "above maximum"
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            return False, "string too short"
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            return False, "string too long"
    return True, None


def validate_action(legal_action: dict | None, action: Any) -> tuple[bool, str | None]:
    """Return (ok, reason) for `action` against the exact `legal_action` schema + distinct rules.

    `reason` mirrors the server's wording (e.g. "speak: string too long", "too few items",
    "a,b must be distinct") so a rejection is self-explanatory in logs."""
    legal_action = legal_action or {}
    ok, err = _validate_json_schema(action, legal_action.get("schema"))
    if not ok:
        return False, err
    for fields in (legal_action.get("rules") or {}).get("distinct", []):
        values = [action.get(f) for f in fields] if isinstance(action, dict) else []
        non_null = [v for v in values if v is not None]
        if len(non_null) != len(set(non_null)):
            return False, f"{','.join(fields)} must be distinct"
    return True, None


def _schema_options(legal_action: dict | None) -> list:
    schema = (legal_action or {}).get("schema") or {}
    return schema.get("oneOf") or schema.get("anyOf") or [schema]


def _required_urgency_enum(legal_action: dict | None) -> list | None:
    """If the discussion speak option requires an `urgency` field, return its allowed enum values.
    Returns None for older schemas that don't require urgency (forward/backward compatible)."""
    for opt in _schema_options(legal_action):
        if isinstance(opt, dict) and "urgency" in (opt.get("required") or []):
            enum = ((opt.get("properties") or {}).get("urgency") or {}).get("enum")
            return list(enum) if enum else [1, 2, 3]
    return None


def fill_required_defaults(action_kind: str, legal_action: dict | None, action: Any) -> Any:
    """Add schema-REQUIRED fields the model commonly omits, when the value is unambiguous. Currently:
    a discussion `speak` whose schema requires `urgency` but where the model omitted it gets the
    lowest allowed urgency. Never overrides a value the model did provide; schema-driven, so it is a
    no-op on older schemas without an urgency requirement."""
    if (action_kind == "onuw.discussion.speak_or_pass" and isinstance(action, dict)
            and isinstance(action.get("speak"), str) and "urgency" not in action):
        enum = _required_urgency_enum(legal_action)
        if enum:
            return {**action, "urgency": enum[0]}
    return action


def normalize_action(action_kind: str, legal_action: dict | None, action: Any) -> Any:
    """The single client-side normalization pipeline applied by every harness arm before validation:
    canonicalize unambiguous encodings -> fill schema-required defaults (urgency) -> clamp lengths.
    Lossless and strategy-neutral; identical for WolfForgeV2 and CharismaBaseline."""
    action = canonicalize_action(action_kind, action)
    action = fill_required_defaults(action_kind, legal_action, action)
    action = clamp_action(action_kind, legal_action, action)
    return action


def _speak_max_length(legal_action: dict | None) -> int | None:
    """Find the maxLength the schema imposes on a discussion `speak` string, if any."""
    schema = (legal_action or {}).get("schema") or {}
    options = schema.get("oneOf") or schema.get("anyOf") or [schema]
    for option in options:
        speak = ((option or {}).get("properties") or {}).get("speak") or {}
        if isinstance(speak, dict) and "maxLength" in speak:
            try:
                return int(speak["maxLength"])
            except (TypeError, ValueError):
                return None
    return None


def canonicalize_action(action_kind: str, action: Any) -> Any:
    """Rewrite a small set of UNAMBIGUOUS equivalent action encodings into the schema's vocabulary.

    Only lossless, unambiguous rewrites are performed — no target guessing, no strategic change:
      {"type": "speak", "text": X} -> {"speak": X}
      {"type": "pass"}             -> {"pass": true}
    Everything else is returned unchanged (and is then validated/repaired/falled-back as usual).
    Applied identically by every harness arm so V2 and CharismaBaseline stay treatment-equivalent."""
    if action_kind == "onuw.discussion.speak_or_pass" and isinstance(action, dict):
        kind = action.get("type")
        if kind == "speak" and isinstance(action.get("text"), str):
            return {"speak": action["text"]}
        if kind == "pass" and "speak" not in action:
            return {"pass": True}
    return action


def _type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, str):
        return "string"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "unknown"


def shape_fingerprint(obj: Any, action: Any) -> dict:
    """A privacy-safe STRUCTURAL fingerprint of a parsed model reply: types, field NAMES, and a
    string length — never speech text, observations, reasoning, or any value content. Used to make a
    shape failure diagnosable in telemetry without storing the raw response."""
    return {
        "parse_succeeded": isinstance(obj, dict) and bool(obj),
        "top_level_type": _type_name(obj),
        "top_level_keys": sorted(map(str, obj.keys())) if isinstance(obj, dict) else None,
        "action_present": isinstance(obj, dict) and "action" in obj,
        "action_type": _type_name(action),
        "action_keys": sorted(map(str, action.keys())) if isinstance(action, dict) else None,
        "speak_length": (len(action["speak"]) if isinstance(action, dict)
                         and isinstance(action.get("speak"), str) else None),
        "pass_value_type": (_type_name(action.get("pass"))
                            if isinstance(action, dict) and "pass" in action else None),
    }


def clamp_action(action_kind: str, legal_action: dict | None, action: Any) -> Any:
    """Salvage an otherwise-good action that only violates a length bound: truncate a discussion
    `speak` to the schema's maxLength so a long-but-valid speech is sent instead of forfeiting the
    turn or burning a repair. Output normalization only — never changes which action is chosen."""
    if (action_kind == "onuw.discussion.speak_or_pass" and isinstance(action, dict)
            and isinstance(action.get("speak"), str)):
        ml = _speak_max_length(legal_action)
        if ml is not None and len(action["speak"]) > ml:
            return {**action, "speak": action["speak"][:ml]}
    return action
