"""The agent-memory scope toggle (ARENA_AGENT_RESET_BETWEEN_GAMES) pinned with a no-LLM probe.

The reference stateful harnesses key their memory by `state_key(turn/event, reset_between_games)`:
  - default (reset between games): every game gets its own state, even within one run;
  - ARENA_AGENT_RESET_BETWEEN_GAMES=0: one state carries across all games in a run, while different
    runs stay separate.

`StateProbeAgent` is a deterministic stand-in for those harnesses — it counts the turns it has seen
under its current key — so these assertions need no network or LLM and exercise the same keying
logic the real harnesses use. Game ids use the production `f"{run_id}_game_{gid:03d}"` shape so the
run_id fallback (split on `_game_`) is exercised even though the wire doesn't carry run_id per-turn.
"""
from __future__ import annotations

import importlib

from examples._harness_util import (
    _run_id_from_game_id, reset_between_games_from_env, state_key, state_path_name,
)
from persuasion_arena_agent.models import Event, Turn

RUN_A = "run_aaa"
RUN_B = "run_bbb"


def _gid(run_id: str, gid: int) -> str:
    return f"{run_id}_game_{gid:03d}"


def _event(run_id: str, gid: int, eid: str = "e") -> Event:
    return Event.from_dict({
        "event_id": eid, "type": "speech", "payload": {"actor_seat": 1, "text": "hi"},
        "game_instance_id": _gid(run_id, gid),
    })


def _turn(run_id: str, gid: int, tid: str = "t") -> Turn:
    return Turn.from_dict({
        "turn_id": tid, "game_instance_id": _gid(run_id, gid), "game": "onuw", "seat": 0,
        "phase": "discussion", "action_kind": "onuw.discussion.speak_or_pass",
        "deadline_at": "2026-06-25T00:00:00Z",
        "observation": {"format": "text", "text": "speak"},
        "legal_action": {"schema": {}, "choices": {"players": [{"seat": 1}, {"seat": 2}]}},
    })


def _fresh_probe(monkeypatch, *, reset: bool | None):
    """Import a fresh StateProbeAgent with the env set BEFORE module import (the module reads env at
    import time to build its module-level default), then return an explicit instance for the test."""
    if reset is None:
        monkeypatch.delenv("ARENA_AGENT_RESET_BETWEEN_GAMES", raising=False)
    else:
        monkeypatch.setenv("ARENA_AGENT_RESET_BETWEEN_GAMES", "1" if reset else "0")
    import examples.state_probe_agent as spa
    importlib.reload(spa)
    return spa.StateProbeAgent()


# -- the pure keying helpers --------------------------------------------------------------------

def test_reset_between_games_from_env_parsing(monkeypatch):
    for raw in ["0", "false", "no", "off", "run", "per-run", "per_run", "FALSE", " Off "]:
        monkeypatch.setenv("ARENA_AGENT_RESET_BETWEEN_GAMES", raw)
        assert reset_between_games_from_env() is False, raw
    for raw in ["1", "true", "yes", "on", "game", "per-game", "per_game", "TRUE", " On "]:
        monkeypatch.setenv("ARENA_AGENT_RESET_BETWEEN_GAMES", raw)
        assert reset_between_games_from_env() is True, raw
    monkeypatch.delenv("ARENA_AGENT_RESET_BETWEEN_GAMES", raising=False)
    assert reset_between_games_from_env() is True          # default
    assert reset_between_games_from_env(default=False) is False
    monkeypatch.setenv("ARENA_AGENT_RESET_BETWEEN_GAMES", "garbage")
    assert reset_between_games_from_env() is True           # unknown -> default
    assert reset_between_games_from_env(default=False) is False


def test_run_id_recovered_from_game_id():
    assert _run_id_from_game_id("run_aaa_game_001") == "run_aaa"
    assert _run_id_from_game_id("game_1") is None           # no separator
    assert _run_id_from_game_id(None) is None


def test_state_key_game_scope_separates_games_in_one_run():
    t1, t2 = _turn(RUN_A, 1), _turn(RUN_A, 2)
    k1 = state_key(t1, reset_between_games=True)
    k2 = state_key(t2, reset_between_games=True)
    assert k1 != k2                                         # each game its own key
    assert k1 == f"{RUN_A}:{_gid(RUN_A, 1)}"


def test_state_key_run_scope_carries_within_run_separates_runs():
    a1 = state_key(_turn(RUN_A, 1), reset_between_games=False)
    a2 = state_key(_turn(RUN_A, 2), reset_between_games=False)
    b1 = state_key(_turn(RUN_B, 1), reset_between_games=False)
    assert a1 == a2 == RUN_A                                # same run -> same key
    assert b1 == RUN_B and a1 != b1                         # different run -> different key


def test_state_key_none_when_no_game_id_in_reset_mode():
    e = Event.from_dict({"event_id": "e", "type": "run_status", "payload": {}})
    assert state_key(e, reset_between_games=True) is None   # run-level event has no game key


def test_state_path_name_is_a_safe_single_segment():
    assert state_path_name("run_aaa:run_aaa_game_001") == "run_aaa_run_aaa_game_001"
    assert state_path_name("///") == "state"
    assert "/" not in state_path_name("a/b\\c d")


# -- the probe harness end-to-end ---------------------------------------------------------------

def test_default_mode_resets_state_between_games(monkeypatch):
    probe = _fresh_probe(monkeypatch, reset=None)           # default == reset between games
    assert probe.reset_between_games is True

    # Game 1: two events + two turns.
    probe.on_event(_event(RUN_A, 1))
    probe.on_event(_event(RUN_A, 1))
    probe.act(_turn(RUN_A, 1))
    out1 = probe.act(_turn(RUN_A, 1))
    assert out1["reasoning"] == (
        f"state_key={RUN_A}:{_gid(RUN_A, 1)}; reset_between_games=True; "
        f"events_seen=2; turns_seen=2"
    )

    # Game 2 in the SAME run starts fresh: no events/turns carried over.
    out2 = probe.act(_turn(RUN_A, 2))
    assert "events_seen=0" in out2["reasoning"]
    assert "turns_seen=1" in out2["reasoning"]
    assert f"state_key={RUN_A}:{_gid(RUN_A, 2)}" in out2["reasoning"]

    # The two games are tracked under distinct keys.
    assert set(probe.turns) == {f"{RUN_A}:{_gid(RUN_A, 1)}", f"{RUN_A}:{_gid(RUN_A, 2)}"}


def test_run_scoped_mode_carries_state_across_games_and_separates_runs(monkeypatch):
    probe = _fresh_probe(monkeypatch, reset=False)          # ARENA_AGENT_RESET_BETWEEN_GAMES=0
    assert probe.reset_between_games is False

    # Run A, game 1 then game 2 — state must accumulate across both games.
    probe.on_event(_event(RUN_A, 1))
    probe.act(_turn(RUN_A, 1))
    probe.on_event(_event(RUN_A, 2))
    out_a2 = probe.act(_turn(RUN_A, 2))
    assert f"state_key={RUN_A}" in out_a2["reasoning"]
    assert "events_seen=2" in out_a2["reasoning"]           # both games' events under one key
    assert "turns_seen=2" in out_a2["reasoning"]            # both turns counted under one key

    # Run B is a separate process scope: starts fresh even with reset off.
    out_b1 = probe.act(_turn(RUN_B, 1))
    assert f"state_key={RUN_B}" in out_b1["reasoning"]
    assert "events_seen=0" in out_b1["reasoning"]
    assert "turns_seen=1" in out_b1["reasoning"]

    assert probe.turns == {RUN_A: 2, RUN_B: 1}
