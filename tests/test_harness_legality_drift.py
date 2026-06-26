"""Drift guards for the example-harness legality layer and the shared JSON parser.

Two things this pins:

1. extract_json (now arena/_jsonparse.py) must behave exactly like the old per-file _extract_json
   that lived (byte-identical) in arena/openrouter.py and examples/_harness_util.py.

2. The example harness re-expresses ONUW legality (instructions / _is_legal / _fallback_action) on
   the client side. The ENGINE is the source of truth: it ships a `legal_action` (schema + choices)
   on every turn. This test harvests EVERY action_kind the engine actually emits in a full game,
   then asserts the harness has real handling for each one — the harness fallback must be a LEGAL
   action under the engine's own shipped legal_action, and discussion must require urgency/stance
   the way the engine now does. If the engine grows a new action_kind or tightens an existing one,
   this fails instead of silently letting the harness drift.
"""
from __future__ import annotations

import json

from arena.games.onuw import ONUW
from examples._harness_util import (
    ACTION_INSTRUCTIONS, action_request, fallback_action, interpret,
)
from examples import _harness_util as hu

NAMES = {i: f"P{i}" for i in range(5)}


class _Turn:
    """Minimal stand-in for persuasion_arena_agent.models.Turn: the harness only reads
    .action_kind and .legal_action off a turn."""

    def __init__(self, action_kind: str, legal_action: dict):
        self.action_kind = action_kind
        self.legal_action = legal_action


class _Recorder:
    """Captures (action_kind, legal_action) for every turn the engine asks it to act on, then
    plays the engine's defaulted action so the game proceeds deterministically."""

    name = "rec"
    model = "scripted"

    def __init__(self) -> None:
        self.turns: list[_Turn] = []

    def act(self, observation, parse_action, default_action, **turn_meta):
        kind = turn_meta.get("action_kind")
        legal = turn_meta.get("legal_action") or {}
        if kind:
            self.turns.append(_Turn(kind, legal))

        class Resp:
            declared_reasoning = "(rec)"
            action = default_action
            ms = 0.0

        return Resp()


def _harvest_engine_turns() -> list[_Turn]:
    """Run a full ONUW game whose deck exercises every night action role plus discussion + vote,
    recording each turn the engine emits."""
    # Deck deliberately includes all action roles so every onuw.* night action_kind is emitted.
    layout = ["Doppelganger", "Seer", "Robber", "Troublemaker", "Drunk", "Werewolf", "Minion", "Villager"]
    core = ONUW(NAMES, seed=3, deck=list(layout), deal_override=list(layout))
    core.deal()
    rec = _Recorder()
    agents = {i: rec for i in range(5)}
    core.run_night(agents)
    core.run_discussion(agents)
    core.run_vote(agents)
    return rec.turns


def test_every_engine_action_kind_has_harness_handling():
    turns = _harvest_engine_turns()
    kinds = {t.action_kind for t in turns}

    # Sanity: the deck above must actually drive the full ONUW action surface.
    expected = {
        "onuw.doppelganger.copy_player",
        "onuw.seer.inspect",
        "onuw.robber.swap_or_decline",
        "onuw.troublemaker.swap_two_or_decline",
        "onuw.drunk.swap_center",
        "onuw.discussion.speak_or_pass",
        "onuw.vote",
    }
    missing = expected - kinds
    assert not missing, f"engine did not emit expected action_kinds: {missing}"

    for t in turns:
        # 1) Every emitted kind has a real (non-generic) instruction in the harness prompt.
        assert t.action_kind in ACTION_INSTRUCTIONS, (
            f"harness has no ACTION_INSTRUCTIONS entry for {t.action_kind} (drift)"
        )
        assert action_request(t).startswith(f"YOUR TURN ({t.action_kind})")

        # 2) The harness fallback must be LEGAL under the engine's OWN shipped legal_action.
        fb = fallback_action(t)
        assert hu._is_legal(t, fb), (
            f"harness fallback {fb!r} is illegal for {t.action_kind} given engine "
            f"legal_action={t.legal_action!r}"
        )


def test_harness_legality_tracks_engine_speech_urgency_requirement():
    """The engine tightened discussion: every speak now REQUIRES an urgency and every pass a
    stance. The harness must not accept a bare {speak:...}/{pass:true} anymore (the stale shape)."""
    turns = _harvest_engine_turns()
    speak = next(t for t in turns if t.action_kind == "onuw.discussion.speak_or_pass")

    assert hu._is_legal(speak, {"speak": "I am the Seer", "urgency": 2})
    assert hu._is_legal(speak, {"pass": True, "stance": "done"})
    # Stale shapes the old harness wrongly accepted:
    assert not hu._is_legal(speak, {"speak": "I am the Seer"})  # missing urgency
    assert not hu._is_legal(speak, {"speak": "hi", "urgency": 5})  # out-of-range urgency

    # A bare string still coerces into a fully-shaped, legal speak.
    coerced_action, _, legal = interpret(speak, json.dumps({"action": "just talking"}))
    assert legal and coerced_action == {"speak": "just talking", "urgency": 1}


def test_vote_fallback_is_no_one_and_legal():
    turns = _harvest_engine_turns()
    vote = next(t for t in turns if t.action_kind == "onuw.vote")
    fb = fallback_action(vote)
    assert fb == {"target": vote.legal_action["choices"]["no_one"]}
    assert hu._is_legal(vote, fb)
    # A real player ref from the shipped choices is also legal.
    ref = vote.legal_action["choices"]["players"][0]["ref"]
    assert hu._is_legal(vote, {"target": ref})


# ---- extract_json parity ---------------------------------------------------


def _old_extract_json(text):
    """The pre-refactor implementation, copied verbatim, to pin behavioral parity."""
    import re

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        frag = m.group(0)
        for end in range(len(frag), 0, -1):
            try:
                return json.loads(frag[:end])
            except json.JSONDecodeError:
                continue
    return None


def test_extract_json_matches_old_extractor():
    from arena._jsonparse import extract_json

    payloads = [
        '{"action": {"pass": true}, "declared_reasoning": "ok"}',
        'Here is my move:\n```json\n{"action": {"speak": "hi", "urgency": 1}}\n```\nthanks',
        'prose {"target": "@p"} trailing garbage }}} not-json',
        'no json at all here',
        '',
        '{"a": 1} and then {"b": 2}',  # first object wins
        '{"nested": {"x": [1, 2, 3]}, "y": "z"}',
    ]
    for p in payloads:
        assert extract_json(p) == _old_extract_json(p), f"parity broke on {p!r}"

    # Both extractors point at the SAME module-level function in arena.openrouter (no second copy).
    import arena.openrouter as orouter

    assert orouter.extract_json is extract_json
