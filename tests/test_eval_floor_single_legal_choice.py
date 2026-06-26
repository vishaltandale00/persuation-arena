"""EVAL-FLOOR: when exactly ONE legal target exists, a competent player MUST take it — and an
out-of-set target MUST be rejected by the real parse/validation code.

This is a deterministic competence FLOOR, scored by SCHEMA-LEGALITY / EXACT-MATCH only (never model
judgment). It asserts on the chosen REF (the action), never on who gets eliminated. Three real code
paths are exercised so the floor cannot be passed by a tautology:

  1. The in-process oracle: examples._harness_util._is_legal + the bare example harnesses
     (pass_agent, role_claiming_agent) on a turn engineered to a single legal player ref.
  2. The engine's own per-kind parse() closures and arena.identity.seat_for_participant_ref:
     the lone candidate resolves to its seat; everything out-of-set raises ValueError("bad target").
  3. The connected store path: store.reply_to_turn validates against the turn's legal_action schema,
     rejecting an out-of-enum target ("invalid_action") and accepting the lone legal one.

The real ONUW legal_action schemas and parse() callbacks are HARVESTED by playing one in-process
game with a recording wrapper that delegates to tests.scripted.ScriptedDefault (which returns the
engine's safe default, so the game completes). We never hand-roll a schema we then validate against
itself — every assertion runs through arena/examples production code.
"""
from __future__ import annotations

import pytest

import examples.pass_agent as pass_agent
import examples.role_claiming_agent as role_claiming_agent
from arena import store
from arena.games.onuw import ONUW
from arena.identity import NO_ONE_REF, participant_refs, seat_for_participant_ref
from examples._harness_util import _is_legal, _legal_players
from persuasion_arena_agent.models import Turn

# Five seats; the explicit 8-card deal (5 players, 3 center) guarantees every night role wakes, so a
# single seeded game surfaces all seven action_kinds.
NAMES = {0: "Alice", 1: "Bob", 2: "Cara", 3: "Dan", 4: "Eve"}
DEAL = ["Doppelganger", "Seer", "Robber", "Troublemaker", "Drunk", "Werewolf", "Werewolf", "Villager"]


class _Recorder:
    """Plays the engine's safe default (via ScriptedDefault) so the game completes, while capturing
    the REAL legal_action schema and the REAL parse() closure the engine hands each role."""

    model = "rec"
    name = "rec"

    def __init__(self, schemas: dict, parsers: dict):
        from tests.scripted import ScriptedDefault

        self._inner = ScriptedDefault()
        self._schemas = schemas
        self._parsers = parsers

    def act(self, observation, parse_action, default_action, **turn_meta):
        kind = turn_meta["action_kind"]
        # First occurrence wins: capture a clean, unmutated schema/closure per kind.
        self._schemas.setdefault(kind, dict(turn_meta["legal_action"]))
        self._parsers.setdefault(kind, parse_action)
        return self._inner.act(observation, parse_action, default_action)


@pytest.fixture(scope="module")
def harvested():
    """(schemas, parsers) keyed by action_kind, harvested from one real game. Module-scoped: the
    engine code is deterministic for this seed/deal, so one play covers every kind."""
    schemas: dict = {}
    parsers: dict = {}
    agents = {i: _Recorder(schemas, parsers) for i in range(len(NAMES))}
    ONUW(NAMES, seed=4242, deal_override=list(DEAL)).play(agents)
    # Sanity: a real game must have exercised every kind we assert on.
    assert "onuw.vote" in schemas and "onuw.robber.swap_or_decline" in parsers
    return schemas, parsers


def _single_choice_vote_turn(lone_seat: int, voter_seat: int = 0) -> Turn:
    """A REAL vote turn whose legal targets are exactly one player ref (+ NO_ONE_REF), built from the
    engine's own schema shape (participant_refs + the enum the engine emits in _vote)."""
    lone_ref = participant_refs(NAMES, [lone_seat])[0]
    return Turn.from_dict({
        "turn_id": "t", "game_instance_id": "g", "game": "onuw", "seat": voter_seat,
        "phase": "vote", "action_kind": "onuw.vote", "deadline_at": "2026-06-25T00:00:00Z",
        "observation": {"format": "text", "text": "vote"},
        "legal_action": {
            "schema": {
                "type": "object", "required": ["target"],
                "properties": {"target": {"type": "string", "enum": [lone_ref, NO_ONE_REF]}},
                "additionalProperties": False,
            },
            "choices": {"players": [{"name": NAMES[lone_seat], "ref": lone_ref}], "no_one": NO_ONE_REF},
        },
    })


# --- 1. Oracle floor: the lone legal target (or NO_ONE_REF) is taken and is schema-legal ----------


def test_single_legal_vote_target_pass_agent_takes_a_legal_choice():
    """pass_agent always abstains; on a one-target vote that abstention (NO_ONE_REF) is still legal."""
    turn = _single_choice_vote_turn(lone_seat=1)
    action = pass_agent.act(turn)["action"]
    assert _is_legal(turn, action) is True
    assert action["target"] in set(_legal_players(turn)) | {NO_ONE_REF}
    assert action == {"target": NO_ONE_REF}  # exact: pass-baseline abstains


def test_single_legal_vote_target_role_claiming_takes_the_lone_player_ref():
    """role_claiming votes the LAST legal player ref; with one player ref that is the lone target."""
    lone_ref = participant_refs(NAMES, [1])[0]
    turn = _single_choice_vote_turn(lone_seat=1)
    action = role_claiming_agent.act(turn)["action"]
    assert _is_legal(turn, action) is True
    assert action == {"target": lone_ref}  # the single legal player ref is chosen exactly


def test_out_of_set_vote_target_is_rejected_by_the_oracle():
    """The oracle must reject a target outside the lone legal set (here a player ref not offered)."""
    turn = _single_choice_vote_turn(lone_seat=1)
    not_offered = participant_refs(NAMES, [2])[0]  # @cara — present in the game but NOT a legal choice
    assert not_offered not in set(_legal_players(turn)) | {NO_ONE_REF}
    assert _is_legal(turn, {"target": not_offered}) is False


# --- 2. parse()/seat-resolution floor: lone candidate resolves; out-of-set raises -----------------


def test_seat_for_participant_ref_single_candidate_resolves_and_rejects_others():
    """With a single candidate seat, only its ref resolves; every other (even a real player) raises."""
    assert seat_for_participant_ref("@bob", NAMES, [1]) == 1
    assert seat_for_participant_ref(1, NAMES, [1]) == 1  # the int seat itself resolves
    for bad in ("@cara", "@dan", "@no-one", "@ghost", 2):
        with pytest.raises(ValueError, match="bad target"):
            seat_for_participant_ref(bad, NAMES, [1])  # allow_no_one defaults False


def test_real_vote_parse_resolves_lone_target_and_rejects_out_of_set(harvested):
    """The engine's own vote parse() closure: the lone offered ref resolves to its seat; NO_ONE_REF
    resolves to the no-kill sentinel; an unknown ref raises."""
    _, parsers = harvested
    parse = parsers["onuw.vote"]
    assert parse({"target": "@bob"}, "") == 1
    assert parse({"target": NO_ONE_REF}, "") == -1  # abstention is the lone always-legal choice
    with pytest.raises(ValueError, match="bad target"):
        parse({"target": "@nobody"}, "")


def test_real_robber_parse_takes_the_only_swap_or_declines_and_rejects_out_of_set(harvested):
    """Robber: the one offered player ref resolves to its seat, decline (target=None) is the other
    always-legal choice, and an out-of-set ref raises."""
    _, parsers = harvested
    parse = parsers["onuw.robber.swap_or_decline"]
    assert parse({"target": "@bob"}, "") == 1
    assert parse({"target": None}, "") is None  # decline
    with pytest.raises(ValueError, match="bad target"):
        parse({"target": "@nobody"}, "")


def test_real_doppelganger_parse_rejects_out_of_set(harvested):
    """Doppelganger has no decline path; a copy target must be an offered player ref or it raises."""
    _, parsers = harvested
    parse = parsers["onuw.doppelganger.copy_player"]
    # The doppelganger is seat 0 (dealt Doppelganger), so it may copy seats 1..4 but never itself.
    assert parse({"target": "@bob"}, "") == 1
    with pytest.raises(ValueError, match="bad target"):
        parse({"target": "@ghost"}, "")
    with pytest.raises(ValueError, match="bad target"):
        parse({"target": "@alice"}, "")  # itself is not a legal copy target


# --- 3. Connected store floor: schema validation accepts the lone target, rejects everything else -


def _sqlite_store(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "single_choice.db")
    store.init_schema()


def test_store_reply_validation_accepts_lone_target_rejects_out_of_set(tmp_path, monkeypatch):
    """store.reply_to_turn validates the action against the turn's legal_action schema. When the enum
    admits exactly one player ref (+ NO_ONE_REF), an out-of-enum target is rejected as
    'invalid_action' and nothing is written; the lone legal target is accepted and recorded."""
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "run_single", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 1, "players": 1, "seed_base": 1,
    })
    agent = store.register_agent("agent", "hash_agent", "arena-agent-v1", "test")
    signup, err = store.create_signup("run_single", agent["id"])
    assert err is None
    store.mark_signup_ready(signup["id"], agent["id"])

    lone_ref = participant_refs(NAMES, [1])[0]  # @bob
    not_offered = participant_refs(NAMES, [2])[0]  # @cara
    turn = store.create_turn(
        "run_single", signup["id"], "game_1", 0, "vote", "onuw.vote",
        {"format": "text", "text": "vote"},
        {"schema": {
            "type": "object", "required": ["target"],
            "properties": {"target": {"type": "string", "enum": [lone_ref, NO_ONE_REF]}},
            "additionalProperties": False,
        }},
    )

    # An out-of-set target is rejected before any writer wins, and leaves no reply.
    _, err = store.reply_to_turn(turn["id"], agent["id"], {"target": not_offered}, "bad")
    assert err == "invalid_action"
    assert store.get_turn_reply(turn["id"]) is None

    # The lone legal target is accepted and recorded exactly.
    reply, err = store.reply_to_turn(turn["id"], agent["id"], {"target": lone_ref}, "good")
    assert err is None
    assert reply["accepted"] == 1
    assert store.get_turn_reply(turn["id"])["action"] == {"target": lone_ref}
