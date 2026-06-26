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
