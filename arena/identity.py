"""Public participant identity helpers.

Game engines keep integer seats internally. Anything shown to agents should use a stable semantic
participant ref derived from the public display name, e.g. "Alice Smith" -> "@alice-smith".
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

MAX_PUBLIC_NAME_LENGTH = 64
NO_ONE_REF = "@no-one"


def normalize_public_name(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def public_handle(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", normalize_public_name(name))
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")


def public_ref(name: str) -> str:
    handle = public_handle(name)
    return f"@{handle}" if handle else ""


def validate_public_name(value: Any) -> tuple[str | None, str | None]:
    name = normalize_public_name(value)
    if not name:
        return None, "display_name required"
    if len(name) > MAX_PUBLIC_NAME_LENGTH:
        return None, f"display_name must be {MAX_PUBLIC_NAME_LENGTH} characters or fewer"
    if not public_handle(name):
        return None, "display_name must contain at least one ASCII letter or number"
    return name, None


def validate_unique_public_names(names: Sequence[str] | Mapping[Any, str]) -> str | None:
    values = list(names.values()) if isinstance(names, Mapping) else list(names)
    seen_names: dict[str, str] = {}
    seen_refs: dict[str, str] = {}
    for raw in values:
        name, err = validate_public_name(raw)
        if err:
            return err
        assert name is not None
        name_key = name.casefold()
        ref_key = public_ref(name).casefold()
        if name_key in seen_names:
            return f"duplicate public participant name: {name}"
        if ref_key in seen_refs:
            return f"ambiguous public participant names: {seen_refs[ref_key]} and {name}"
        seen_names[name_key] = name
        seen_refs[ref_key] = name
    return None


def participant(name: str) -> dict[str, str]:
    clean = normalize_public_name(name)
    return {"name": clean, "ref": public_ref(clean)}


def participants_for(names: Mapping[int, str]) -> dict[int, dict[str, str]]:
    return {int(seat): participant(name) for seat, name in names.items()}


def participant_label(names: Mapping[int, str], seat: int) -> str:
    p = participant(names[seat])
    return f"{p['name']} ({p['ref']})"


def roster_line(names: Mapping[int, str]) -> str:
    return ", ".join(participant_label(names, i) for i in range(len(names)))


def participant_choices(names: Mapping[int, str], seats: Sequence[int]) -> list[dict[str, str]]:
    return [participant(names[i]) for i in seats]


def participant_refs(names: Mapping[int, str], seats: Sequence[int]) -> list[str]:
    return [public_ref(names[i]) for i in seats]


def _ref_lookup(names: Mapping[int, str], candidates: Sequence[int]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for seat in candidates:
        name = normalize_public_name(names[seat])
        ref = public_ref(name)
        handle = public_handle(name)
        lookup[ref.casefold()] = seat
        lookup[handle.casefold()] = seat
        lookup[name.casefold()] = seat
    return lookup


def seat_for_participant_ref(value: Any, names: Mapping[int, str], candidates: Sequence[int], *,
                             allow_no_one: bool = False) -> int:
    if allow_no_one and value in (-1, NO_ONE_REF):
        return -1
    if isinstance(value, int) and not isinstance(value, bool):
        if value in candidates:
            return value
        raise ValueError("bad target")
    if not isinstance(value, str):
        raise ValueError("bad target")
    raw = value.strip()
    if allow_no_one and raw.casefold() in {"-1", "none", "no one", "no_one", NO_ONE_REF}:
        return -1
    lookup = _ref_lookup(names, candidates)
    key = raw.casefold()
    if key in lookup:
        return lookup[key]
    if not raw.startswith("@") and f"@{key}" in lookup:
        return lookup[f"@{key}"]
    raise ValueError("bad target")
