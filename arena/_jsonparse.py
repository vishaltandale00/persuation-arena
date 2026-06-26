"""Tolerant JSON extraction shared by the arena's OpenRouter client and the example harnesses.

Models wrap their JSON in code fences or surrounding prose. `extract_json` grabs the first
balanced-looking {...} block and parses it, trimming trailing junk if the tail is malformed.
This lived (byte-identical) in both arena/openrouter.py and examples/_harness_util.py; it now
lives here so the two cannot drift.
"""
from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict | None:
    # tolerate code fences / surrounding prose: grab the first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        m = re.search(r"\{.*", text, re.DOTALL)
    if not m:
        return None
    frag = m.group(0)
    candidates = [frag]
    closers = _json_closers_for_fragment(frag)
    if closers:
        candidates.append(frag + closers)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # try trimming trailing junk
            for end in range(len(candidate), 0, -1):
                prefix = candidate[:end]
                repair = _json_closers_for_fragment(prefix)
                if repair:
                    prefix += repair
                try:
                    return json.loads(prefix)
                except json.JSONDecodeError:
                    continue
    return None


def _json_closers_for_fragment(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    if in_string:
        return ""
    return "".join(reversed(stack))
