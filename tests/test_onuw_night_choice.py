"""Focused tests for the extracted ONUW `_night_choice` helper.

The helper de-duplicates the boilerplate shared by the five single-decision night roles
(Doppelganger/Seer/Robber/Troublemaker/Drunk): night `base_prompt` + inline-JSON suffix, the
`_act` call wired with the night phase and the role's `action_kind`, and a `legal_action` carrying
the schema, choices, and a `default_wire_action`. These tests pin that wiring so the refactor stays
behavior-preserving: every night role must route through `_night_choice`, and the turn_meta it hands
each agent (phase / action_kind / legal_action / default_wire_action) must match what each role
declares. `_vote` must NOT route through the helper (it uses the async prepare/wait contract).
"""
from __future__ import annotations

from arena.games.onuw import ONUW

NAMES = {i: f"P{i}" for i in range(5)}

# 8-card layout (5 players + 3 center) that wakes every single-decision night role and routes the
# Doppelganger through the Seer copy-action, so one deal exercises the whole helper-backed set.
LAYOUT = ["Doppelganger", "Seer", "Robber", "Troublemaker", "Drunk", "Werewolf", "Minion", "Mason"]

# Map each night role's action_kind to the engine method that should drive it through the helper.
NIGHT_ROLES = {
    "onuw.doppelganger.copy_player": ("_doppelganger", 0),
    "onuw.seer.inspect": ("_seer_action", 1),
    "onuw.robber.swap_or_decline": ("_robber_action", 2),
    "onuw.troublemaker.swap_two_or_decline": ("_tm_action", 3),
    "onuw.drunk.swap_center": ("_drunk_action", 4),
}


class CapturingAgent:
    """Records every turn_meta + prompt it sees and always returns the safe default."""
    name = "Cap"
    model = "scripted"

    def __init__(self):
        self.calls: list[dict] = []

    def act(self, observation, parse_action, default_action, **turn_meta):
        self.calls.append({
            "observation": observation,
            "default_action": default_action,
            "turn_meta": turn_meta,
        })

        class Resp:
            declared_reasoning = "(cap)"
            action = default_action
            ms = 0.0

        return Resp()


def _fresh_core():
    core = ONUW(NAMES, seed=3, deck=list(LAYOUT), deal_override=list(LAYOUT))
    core.deal()
    return core


def test_night_choice_wires_turn_meta_for_each_role():
    """Each role's method, driven through `_night_choice`, must hand the agent night-phase
    turn_meta with the role's own action_kind, a non-self default, and a legal_action whose
    schema/choices/default_wire_action are present and consistent."""
    for action_kind, (method_name, seat) in NIGHT_ROLES.items():
        core = _fresh_core()
        agent = CapturingAgent()
        getattr(core, method_name)(seat, agent)

        # The role method calls the agent exactly once for its own decision (Doppelganger
        # additionally performs the copied action, but seat 0 copies the Seer which then calls
        # again — assert the FIRST call is the role's own decision).
        first = agent.calls[0]
        meta = first["turn_meta"]
        assert meta["phase"] == "night", action_kind
        assert meta["action_kind"] == action_kind
        # prompt is the night base_prompt plus the inline-JSON action suffix
        assert "NIGHT ACTION" in first["observation"]
        assert "ONUW_RULES_PAYLOAD_BEGIN" in first["observation"]
        # legal_action carries schema + choices + a default_wire_action
        legal = meta["legal_action"]
        assert "schema" in legal and "choices" in legal
        assert isinstance(meta["default_wire_action"], dict) and meta["default_wire_action"]
        # default action never targets the acting seat
        assert first["default_action"] != seat


def test_troublemaker_keeps_distinct_rule_through_helper():
    """Only the Troublemaker passes a `rules` block; the helper must forward it, and the other
    roles must NOT gain a spurious `rules` key."""
    core = _fresh_core()
    agent = CapturingAgent()
    core._tm_action(3, agent)
    legal = agent.calls[0]["turn_meta"]["legal_action"]
    assert legal["rules"] == {"distinct": [["a", "b"]]}

    for method_name, seat in (("_robber_action", 2), ("_drunk_action", 4)):
        c = _fresh_core()
        a = CapturingAgent()
        getattr(c, method_name)(seat, a)
        assert "rules" not in a.calls[0]["turn_meta"]["legal_action"]


def test_seer_default_wire_action_is_valid_center_view():
    """The Seer's default_wire_action must be a concrete legal center view (the documented safe
    fallback), proving the helper forwards the role-specific wire default verbatim."""
    core = _fresh_core()
    agent = CapturingAgent()
    core._seer_action(1, agent)
    wire = agent.calls[0]["turn_meta"]["default_wire_action"]
    assert wire == {"mode": "center", "indices": [0, 1]}


def test_robber_schema_reuses_participant_target_schema_nullable():
    """The Robber routes its schema through `_participant_target_schema(..., nullable=True)`; the
    helper must forward exactly that object so 'decline' (null target) stays representable."""
    core = _fresh_core()
    expected = core._participant_target_schema("target", [i for i in range(5) if i != 2], nullable=True)
    agent = CapturingAgent()
    core._robber_action(2, agent)
    assert agent.calls[0]["turn_meta"]["legal_action"]["schema"] == expected


def test_vote_does_not_route_through_night_choice():
    """`_vote` uses the async prepare/wait contract and must stay OUT of the single-shot helper:
    its turn_meta carries the vote phase, not a night action_kind."""
    core = _fresh_core()
    agent = CapturingAgent()
    core._vote(0, agent, frozen_public=[], wait=True)
    meta = agent.calls[0]["turn_meta"]
    assert meta["phase"] == "vote"
    assert meta["action_kind"] == "onuw.vote"
    # the vote kind is NOT one of the helper-backed night kinds
    assert meta["action_kind"] not in NIGHT_ROLES
