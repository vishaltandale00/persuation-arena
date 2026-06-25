"""Regression tests for the connected-run 422 class: client-side validation must match the server's
exact legal-action schema, and 422s must surface safe detail. No network, no paid model calls.

Root failure reproduced: a generic/connected harness submitted an action that passed its loose local
check but failed arena.store._validate_action server-side (e.g. an over-long discussion speech, or a
Seer center-inspect with too few / non-distinct indices), producing a 422 on /reply.
"""
from __future__ import annotations

import json

import httpx

from arena import store
from arena.games.onuw import ONUW
from examples import _harness_util as hu
from examples import random_agent
from examples._action_schema import clamp_action, normalize_action, validate_action
from persuasion_arena_agent import ArenaApiError
from persuasion_arena_agent.client import ArenaHttpClient
from persuasion_arena_agent.credentials import AgentCredentials

NAMES = {i: f"P{i}" for i in range(5)}


# --- exact legal-action payloads, taken straight from the engine -----------------------------------

def _engine_legal(action_kind: str) -> dict:
    """Build the EXACT legal_action the ONUW engine serves for a given action kind (seat 0)."""
    core = ONUW(NAMES, seed=1)
    core.deal()
    captured = {}

    class Capture:
        name = "C"
        model = "scripted"

        def act(self, observation, parse_action, default_action, **meta):
            captured["legal_action"] = meta["legal_action"]

            class R:
                declared_reasoning = ""
                action = default_action
                ms = 0.0
            return R()

    agent = Capture()
    if action_kind == "onuw.seer.inspect":
        core._seer_action(0, agent)
    elif action_kind == "onuw.discussion.speak_or_pass":
        core._speech_bid(0, agent)
    elif action_kind == "onuw.troublemaker.swap_two_or_decline":
        core._tm_action(0, agent)
    elif action_kind == "onuw.robber.swap_or_decline":
        core._robber_action(0, agent)
    elif action_kind == "onuw.vote":
        core._vote(0, agent, [], wait=False)
    else:
        raise AssertionError(action_kind)
    return captured["legal_action"]


class _Turn:
    """Minimal stand-in for the SDK Turn (only the fields _harness_util.interpret reads)."""

    def __init__(self, action_kind, legal_action):
        self.action_kind = action_kind
        self.legal_action = legal_action


# --- 1 + 2: Seer center inspection index-count rules match the server -------------------------------

def test_seer_center_one_index_rejected_two_accepted():
    legal = _engine_legal("onuw.seer.inspect")
    # one index: loose local check historically accepted it; the server schema requires exactly two.
    assert store._validate_action({"mode": "center", "indices": [0]}, legal)[0] is False
    ok_one, _, legal_one = hu.interpret(_Turn("onuw.seer.inspect", legal),
                                        json.dumps({"action": {"mode": "center", "indices": [0]}}))
    assert legal_one is False                      # canonical validator now rejects it client-side
    # two distinct indices: accepted by both server and client
    assert store._validate_action({"mode": "center", "indices": [0, 1]}, legal)[0] is True
    _, _, legal_two = hu.interpret(_Turn("onuw.seer.inspect", legal),
                                   json.dumps({"action": {"mode": "center", "indices": [0, 1]}}))
    assert legal_two is True


def test_seer_center_duplicate_indices_rejected():
    legal = _engine_legal("onuw.seer.inspect")
    assert store._validate_action({"mode": "center", "indices": [1, 1]}, legal)[0] is False
    _, _, legal = hu.interpret(_Turn("onuw.seer.inspect", legal),
                               json.dumps({"action": {"mode": "center", "indices": [1, 1]}}))
    assert legal is False


# --- 3: additionalProperties forbidden -------------------------------------------------------------

def test_additional_properties_rejected():
    legal = _engine_legal("onuw.discussion.speak_or_pass")
    bad = {"speak": "hello", "extra": 1}
    assert store._validate_action(bad, legal)[0] is False
    _, _, legal_b = hu.interpret(_Turn("onuw.discussion.speak_or_pass", legal),
                                 json.dumps({"action": bad}))
    assert legal_b is False


# --- 4: troublemaker distinct rule matches the server ----------------------------------------------

def test_troublemaker_distinct_rule_matches_server():
    legal = _engine_legal("onuw.troublemaker.swap_two_or_decline")
    assert "distinct" in (legal.get("rules") or {})
    # same-participant pair is rejected by both (refs, post seat->@ref migration)
    assert store._validate_action({"a": "@p1", "b": "@p1"}, legal)[0] is False
    assert validate_action(legal, {"a": "@p1", "b": "@p1"})[0] is False
    # distinct pair and the decline form are accepted by both
    assert store._validate_action({"a": "@p1", "b": "@p2"}, legal)[0] is True
    assert validate_action(legal, {"a": "@p1", "b": "@p2"})[0] is True
    assert store._validate_action({"a": None, "b": None}, legal)[0] is True
    assert validate_action(legal, {"a": None, "b": None})[0] is True


# --- the canary failure: over-long discussion speak ------------------------------------------------

def test_generic_harness_canonicalizes_type_speak():
    # The generic SessionAgent path must apply the same canonicalization as V2 (treatment parity):
    # {"type":"speak","text":X} -> {"speak":X}, accepted and server-valid.
    legal = _engine_legal("onuw.discussion.speak_or_pass")
    action, _, legal_flag = hu.interpret(_Turn("onuw.discussion.speak_or_pass", legal),
                                         json.dumps({"action": {"type": "speak", "text": "hi table"}}))
    # canonicalized to {"speak":...} and the schema-required urgency is filled in -> server-valid
    assert legal_flag is True and action == {"speak": "hi table", "urgency": 1}
    assert store._validate_action(action, legal)[0] is True


def test_over_long_speak_rejected_by_server_and_clamped_client_side():
    legal = _engine_legal("onuw.discussion.speak_or_pass")
    # server rejects an over-long speak (this is the wf_v2_canary_20260625_114536 422)
    ok, reason = store._validate_action({"speak": "x" * 1500, "urgency": 1}, legal)
    assert ok is False and "too long" in reason
    # clamp_action truncates to maxLength; normalize_action also fills the required urgency -> valid
    clamped = clamp_action("onuw.discussion.speak_or_pass", legal, {"speak": "x" * 1500})
    assert len(clamped["speak"]) == 1000
    fixed = normalize_action("onuw.discussion.speak_or_pass", legal, {"speak": "x" * 1500})
    assert len(fixed["speak"]) == 1000 and fixed["urgency"] in (1, 2, 3)
    assert store._validate_action(fixed, legal)[0] is True
    # end-to-end through the generic harness interpret(): long speak comes back legal + clamped
    action, _, legal_flag = hu.interpret(_Turn("onuw.discussion.speak_or_pass", legal),
                                         json.dumps({"action": {"speak": "x" * 1500}}))
    assert legal_flag is True and len(action["speak"]) == 1000


# --- 5 + 6: generic SessionAgent decide() repairs to / falls back to an exactly-valid action -------

def test_decide_repair_and_fallback_are_exactly_valid(monkeypatch):
    legal = _engine_legal("onuw.seer.inspect")
    turn = _Turn("onuw.seer.inspect", legal)

    # First model reply illegal (one index), repair returns a valid two-index inspect.
    replies = iter([
        {"role": "assistant", "content": json.dumps({"reasoning": "r", "action": {"mode": "center", "indices": [0]}})},
        {"role": "assistant", "content": json.dumps({"reasoning": "r", "action": {"mode": "center", "indices": [0, 2]}})},
    ])
    monkeypatch.setattr(hu, "_llm", lambda *a, **k: next(replies))
    action, _, _ = hu.decide("fake/model", [{"role": "user", "content": "x"}], turn)
    assert action == ("center", [0, 2]) or action == {"mode": "center", "indices": [0, 2]}
    # whatever shape parse_action returns, the raw chosen action must be server-valid
    assert store._validate_action({"mode": "center", "indices": [0, 2]}, legal)[0] is True

    # Both replies illegal -> deterministic fallback, which must also pass the exact validator.
    bad = iter([
        {"role": "assistant", "content": json.dumps({"action": {"mode": "center", "indices": [0]}})},
        {"role": "assistant", "content": json.dumps({"action": {"mode": "center", "indices": [1]}})},
    ])
    monkeypatch.setattr(hu, "_llm", lambda *a, **k: next(bad))
    fb = hu.fallback_action(turn)
    assert store._validate_action(fb, legal)[0] is True


def test_every_kind_fallback_is_server_valid():
    for kind in ("onuw.discussion.speak_or_pass", "onuw.vote", "onuw.seer.inspect",
                 "onuw.robber.swap_or_decline", "onuw.troublemaker.swap_two_or_decline"):
        legal = _engine_legal(kind)
        fb = hu.fallback_action(_Turn(kind, legal))
        ok, reason = store._validate_action(fb, legal)
        assert ok is True, f"{kind} fallback {fb} rejected: {reason}"


# --- the fake-brain opponent path: every action it submits is server-valid (run wf_v2_refs_fake_…) --

def test_reference_opponent_actions_valid_for_every_kind():
    # The fake-brain opponent (examples/random_agent) must emit only server-valid actions under the
    # CURRENT protocol. Repeat many times to cover its random branches.
    for kind in ("onuw.discussion.speak_or_pass", "onuw.vote", "onuw.seer.inspect",
                 "onuw.robber.swap_or_decline", "onuw.troublemaker.swap_two_or_decline"):
        legal = _engine_legal(kind)
        turn = _Turn(kind, legal)
        for _ in range(40):
            action = random_agent.act(turn)["action"]
            ok, reason = store._validate_action(action, legal)
            assert ok, f"{kind}: reference opponent submitted invalid {action}: {reason}"
            # never an integer seat or the legacy -1 sentinel
            assert not isinstance(action.get("target"), int)


def test_old_discussion_speak_missing_urgency_rejected_and_not_emitted():
    # Exact wf_v2_refs_fake_20260625_175134 failure: a discussion speak WITHOUT urgency is invalid.
    legal = _engine_legal("onuw.discussion.speak_or_pass")
    assert store._validate_action({"speak": "hi"}, legal)[0] is False           # server rejects
    assert validate_action(legal, {"speak": "hi"})[0] is False                  # client rejects too
    # and the reference opponent NEVER emits that stale shape — speak carries urgency, pass is structured
    turn = _Turn("onuw.discussion.speak_or_pass", legal)
    for _ in range(40):
        a = random_agent.act(turn)["action"]
        assert store._validate_action(a, legal)[0] is True
        assert ("speak" in a and isinstance(a.get("urgency"), int)) or a.get("pass") is True


# --- parity: client validator never disagrees with the server's on these cases ---------------------

def test_client_validator_matches_server_battery():
    cases = {
        "onuw.discussion.speak_or_pass": [
            {"speak": "ok"}, {"pass": True}, {"speak": ""}, {"speak": "y" * 2000},
            {"pass": False}, {"speak": "ok", "x": 1}, {},
        ],
        "onuw.seer.inspect": [
            {"mode": "center", "indices": [0, 1]}, {"mode": "center", "indices": [0]},
            {"mode": "center", "indices": [0, 0]}, {"mode": "player", "target": "@p1"},
            {"mode": "player", "target": "@p99"}, {"mode": "bogus"},
        ],
        "onuw.troublemaker.swap_two_or_decline": [
            {"a": "@p1", "b": "@p2"}, {"a": "@p1", "b": "@p1"}, {"a": None, "b": None},
            {"a": "@p1", "b": "@p99"},
        ],
        "onuw.robber.swap_or_decline": [
            {"target": "@p1"}, {"target": None}, {"target": "@p99"}, {},
        ],
        "onuw.vote": [{"target": "@p1"}, {"target": "@no-one"}, {"target": "@p0"},
                      {"target": "@p99"}, {}],
    }
    for kind, actions in cases.items():
        legal = _engine_legal(kind)
        for action in actions:
            server_ok = store._validate_action(action, legal)[0]
            client_ok = validate_action(legal, action)[0]
            assert server_ok == client_ok, f"{kind} {action}: server={server_ok} client={client_ok}"


# --- 7: no invalid action reaches /reply through a fake transport ----------------------------------

def test_invalid_action_never_submitted_via_fake_transport():
    legal = _engine_legal("onuw.discussion.speak_or_pass")
    submitted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/turns/turn_1/reply":
            body = json.loads(request.content or b"{}")
            ok, reason = store._validate_action(body["action"], legal)
            submitted.append((body["action"], ok))
            if not ok:
                return httpx.Response(422, json={"detail": f"invalid action: {reason}"})
            return httpx.Response(200, json={"ok": True, "accepted": True})
        return httpx.Response(404)

    client = ArenaHttpClient("https://x.test", transport=httpx.MockTransport(handler))
    creds = AgentCredentials("https://x.test", "a", "n", "tok")

    # The harness decided on a clamped (valid) speak — it must be accepted, not 422.
    action, _, legal_flag = hu.interpret(_Turn("onuw.discussion.speak_or_pass", legal),
                                         json.dumps({"action": {"speak": "z" * 1500}}))
    assert legal_flag is True
    client.reply_turn(creds, "turn_1", action, "r", 1)
    assert submitted and submitted[-1][1] is True   # what we submitted was server-valid


# --- 8: 422 exposes safe validation detail, never credentials --------------------------------------

def test_422_surfaces_safe_detail_without_credentials():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/turns/turn_1/reply":
            return httpx.Response(422, json={"detail": "invalid action: speak: string too long"})
        return httpx.Response(404)

    client = ArenaHttpClient("https://x.test", transport=httpx.MockTransport(handler))
    creds = AgentCredentials("https://x.test", "a", "n", "super_secret_token")
    try:
        client.reply_turn(creds, "turn_1", {"speak": "x" * 1500}, "r", 1)
        raise AssertionError("expected ArenaApiError")
    except ArenaApiError as e:
        assert e.status_code == 422
        assert "string too long" in e.detail        # actionable validation reason present
        assert "super_secret_token" not in str(e) and "super_secret_token" not in e.detail
        assert "Bearer" not in str(e)


# --- 9: runner/store diagnostics show safe sqlite path or "postgres", never the URL ----------------

def test_active_backend_label_is_safe(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    label = store.active_backend_label()
    assert label.startswith("sqlite:") and label.endswith("arena.db")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@host/db")
    label_pg = store.active_backend_label()
    assert label_pg == "postgres"                   # never the URL
    assert "secret" not in label_pg and "postgresql://" not in label_pg
