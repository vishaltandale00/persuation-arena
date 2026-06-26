"""EVAL FLOOR: every bundled no-API harness, on a well-formed turn of EVERY ONUW action_kind,
returns a SCHEMA-LEGAL action and NEVER forfeits.

This is a competence FLOOR — a deterministic bar any competent player MUST clear 100%. Scoring is
EXACT-MATCH / SCHEMA-LEGALITY only (the real engine oracle `_is_legal`), never model judgment.

It is not a tautology: the legality verdict comes from the SAME `examples._harness_util._is_legal`
the real harnesses/server use, and the per-kind `legal_action` payloads carried on each synthetic
Turn are HARVESTED from the real engine by playing one in-process ONUW game (no hand-rolled
schemas). So a harness that emitted an illegal action would actually fail here.

Free + hermetic: no network, no API key, no real LLM. The two stateful harnesses (session_agent,
file_memory_agent) route their brain through a module-level `decide(...)`; we monkeypatch that
seam where it is USED to return a guaranteed-legal `fallback_action(turn)`, so they run keyless.
"""
from __future__ import annotations

import random

import pytest

from arena.games.onuw import ONUW
from arena.identity import NO_ONE_REF
from examples._harness_util import _is_legal, fallback_action
from persuasion_arena_agent.models import Turn
from tests.scripted import ScriptedDefault

# The 7 ONUW action kinds the engine can ask any seat to act on.
ACTION_KINDS = [
    "onuw.discussion.speak_or_pass",
    "onuw.vote",
    "onuw.seer.inspect",
    "onuw.robber.swap_or_decline",
    "onuw.troublemaker.swap_two_or_decline",
    "onuw.drunk.swap_center",
    "onuw.doppelganger.copy_player",
]

GID = "floor_game"

# A deal that wakes every night role at least once, so playing one game surfaces a real
# `legal_action` for all six night kinds plus discussion + vote.
_DEAL_OVERRIDE = ["Doppelganger", "Seer", "Robber", "Troublemaker", "Drunk",
                  "Werewolf", "Werewolf", "Villager"]
_NAMES = {0: "Alice", 1: "Bob", 2: "Carol", 3: "Dave", 4: "Eve", 5: "Frank"}


class _Recorder:
    """Plays VALID actions (delegates to ScriptedDefault so the game completes) while capturing the
    real per-kind legal_action the engine hands each seat."""
    name = "R"
    model = "scripted"

    def __init__(self, sink: dict):
        self._inner = ScriptedDefault()
        self._sink = sink

    def act(self, observation, parse_action, default_action, **turn_meta):
        kind = turn_meta.get("action_kind")
        legal = turn_meta.get("legal_action")
        if kind and legal is not None and kind not in self._sink:
            self._sink[kind] = legal
        return self._inner.act(observation, parse_action, default_action)


@pytest.fixture(scope="module")
def real_legal_actions() -> dict[str, dict]:
    """Harvest the engine's real `legal_action` for every action_kind by playing ONE in-process
    game. Asserts against the real engine output — not a literal we typed."""
    sink: dict[str, dict] = {}
    agents = {i: _Recorder(sink) for i in _NAMES}
    ONUW(dict(_NAMES), seed=4242, deal_override=_DEAL_OVERRIDE).play(agents)
    missing = [k for k in ACTION_KINDS if k not in sink]
    assert not missing, f"engine never surfaced these action kinds to harvest: {missing}"
    return sink


def _turn_for(kind: str, legal_action: dict) -> Turn:
    """A well-formed Turn of `kind` carrying the engine's REAL legal_action (same shape as
    tests/test_coding_agent_harness.py::_speak_turn)."""
    return Turn.from_dict({
        "turn_id": f"t-{kind}", "game_instance_id": GID, "game": "onuw", "seat": 0,
        "phase": "discussion" if kind == "onuw.discussion.speak_or_pass" else "night",
        "action_kind": kind,
        "deadline_at": "2026-06-23T17:00:00Z",
        "observation": {"format": "text", "text": "your turn"},
        "legal_action": legal_action,
    })


# ---- the harnesses under test -------------------------------------------------------------------

def _bare_act(module_name: str):
    """Import a bare-act harness's module-level `act` lazily so a single broken import can't take
    out the whole parametrization at collection time."""
    import importlib
    return importlib.import_module(module_name).act


BARE_HARNESSES = [
    "examples.pass_agent",
    "examples.random_agent",
    "examples.role_claiming_agent",
    "examples.adversarial_agent",
    "examples.noisy_talker",
]

STATEFUL_HARNESSES = [
    "examples.session_agent",
    "examples.file_memory_agent",
]


def _assert_floor(out, turn) -> None:
    """The floor itself: a dict, a real action (not empty/None), and schema-legal per the engine."""
    assert isinstance(out, dict), f"harness returned non-dict: {out!r}"
    action = out.get("action")
    assert action not in ({}, None), f"harness forfeited / produced no action: {out!r}"
    assert _is_legal(turn, action) is True, f"illegal action for {turn.action_kind}: {action!r}"


@pytest.mark.parametrize("module_name", BARE_HARNESSES)
@pytest.mark.parametrize("kind", ACTION_KINDS)
def test_bare_harness_is_legal_on_every_kind(module_name, kind, real_legal_actions):
    act = _bare_act(module_name)
    turn = _turn_for(kind, real_legal_actions[kind])
    _assert_floor(act(turn), turn)


def test_random_agent_is_legal_on_both_branches_across_seeds(real_legal_actions):
    """random_agent flips between speak/pass (and random targets) off a module-global RNG. Legality
    must hold on EVERY branch, so we drive it across many seeds for each kind and assert all clear
    the floor. We at least observe both discussion branches to prove both are exercised."""
    import examples.random_agent as ra
    saw_speak = saw_pass = False
    for kind in ACTION_KINDS:
        turn = _turn_for(kind, real_legal_actions[kind])
        for seed in range(40):
            ra._rng = random.Random(seed)
            out = ra.act(turn)
            _assert_floor(out, turn)
            if kind == "onuw.discussion.speak_or_pass":
                if "speak" in out["action"]:
                    saw_speak = True
                if out["action"].get("pass") is True:
                    saw_pass = True
    assert saw_speak and saw_pass, "random_agent never exercised both speak and pass branches"


@pytest.mark.parametrize("module_name", STATEFUL_HARNESSES)
@pytest.mark.parametrize("kind", ACTION_KINDS)
def test_stateful_harness_is_legal_on_every_kind(module_name, kind, real_legal_actions, monkeypatch):
    """The stateful harnesses call an LLM through `decide`. Mock that seam (patched where it is
    USED) to return a guaranteed-legal action so they run keyless — the harness's own memory
    plumbing (event rendering, session/file I/O, the act() wrapper) still runs for real."""
    import importlib
    mod = importlib.import_module(module_name)

    def fake_decide(model, messages, turn):
        # decide(...) -> (action, reasoning, assistant_message). Return a legal action and an
        # appendable assistant dict (session_agent stores it back in the live chat session).
        return fallback_action(turn), "floor", {"role": "assistant", "content": "floor"}

    monkeypatch.setattr(f"{module_name}.decide", fake_decide)

    turn = _turn_for(kind, real_legal_actions[kind])
    out = mod.act(turn)
    _assert_floor(out, turn)


# A clearly-illegal action per kind — an out-of-enum target / index, or a non-dict — that the
# oracle MUST reject. (Robber/Troublemaker treat a null target as a legal "decline", so a bogus
# dict isn't enough there; we use an unknown @ref, which is not in `choices.players`.)
_ILLEGAL_ACTION = {
    "onuw.discussion.speak_or_pass": {"speak": ""},                 # empty speech is not legal
    "onuw.vote": {"target": "@nobody"},                             # not a candidate, not @no-one
    "onuw.seer.inspect": {"mode": "player", "target": "@nobody"},   # unknown target
    "onuw.robber.swap_or_decline": {"target": "@nobody"},           # unknown ref (not None)
    "onuw.troublemaker.swap_two_or_decline": {"a": "@nobody", "b": "@nobody"},  # unknown + non-distinct
    "onuw.drunk.swap_center": {"index": 9},                         # out-of-range center index
    "onuw.doppelganger.copy_player": {"target": "@nobody"},         # unknown target
}


def test_harvested_legal_actions_are_real_engine_output(real_legal_actions):
    """Guardrail against silent tautology: every harvested payload must be a real legal_action dict
    with the choices the oracle reads; the engine's own fallback must clear the oracle; and a
    clearly-illegal action must FAIL it (so the oracle isn't trivially true for everything)."""
    for kind in ACTION_KINDS:
        la = real_legal_actions[kind]
        assert isinstance(la, dict) and "schema" in la and "choices" in la
        turn = _turn_for(kind, la)
        # The engine's guaranteed-legal fallback must itself pass the oracle for this kind.
        assert _is_legal(turn, fallback_action(turn)) is True
        # A deliberately-illegal action must FAIL the oracle.
        assert _is_legal(turn, _ILLEGAL_ACTION[kind]) is False, kind
        # A non-dict is never legal for any kind.
        assert _is_legal(turn, "not a dict") is False, kind
    # Sanity: vote accepts the abstention ref via the engine's NO_ONE_REF branch.
    vote_turn = _turn_for("onuw.vote", real_legal_actions["onuw.vote"])
    assert _is_legal(vote_turn, {"target": NO_ONE_REF}) is True
