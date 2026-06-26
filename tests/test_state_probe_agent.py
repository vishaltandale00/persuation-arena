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
from persuasion_arena_agent.models import Event, PollResponse, Turn

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
    # codex PR#27 r2: game scope is the raw (already-unique) game id, NOT run_id-prefixed —
    # re-prefixing would double an uncapped run id and risk the 255-byte filename limit.
    assert k1 == _gid(RUN_A, 1)


def test_state_key_run_scope_carries_within_run_separates_runs():
    a1 = state_key(_turn(RUN_A, 1), reset_between_games=False)
    a2 = state_key(_turn(RUN_A, 2), reset_between_games=False)
    b1 = state_key(_turn(RUN_B, 1), reset_between_games=False)
    assert a1 == a2 == RUN_A                                # same run -> same key
    assert b1 == RUN_B and a1 != b1                         # different run -> different key


def test_state_key_none_when_no_game_id_in_reset_mode():
    e = Event.from_dict({"event_id": "e", "type": "run_status", "payload": {}})
    assert state_key(e, reset_between_games=True) is None   # run-level event has no game key


def test_pollresponse_threads_run_id_onto_events_and_turn():
    """codex PR#27 P2: the poll envelope carries run_id at the top level; PollResponse must thread it
    onto child Event/Turn objects so run-scoped keying works for ANY game-id scheme."""
    pr = PollResponse.from_dict({
        "signup_id": "s1", "run_id": RUN_A, "run_status": "active", "poll_after_ms": 100,
        "events": [{"event_id": "e1", "type": "speech", "payload": {}, "game_instance_id": "game_1"}],
        "turn": {
            "turn_id": "t1", "game_instance_id": "game_1", "game": "onuw", "seat": 0,
            "phase": "discussion", "action_kind": "onuw.discussion.speak_or_pass",
            "deadline_at": "2026-06-25T00:00:00Z", "observation": {}, "legal_action": {},
        },
    })
    assert pr.events[0].run_id == RUN_A                    # inherited from the envelope
    assert pr.turn.run_id == RUN_A
    # And a per-row run_id, when present, is preserved (not clobbered by the envelope).
    pr2 = PollResponse.from_dict({
        "signup_id": "s1", "run_id": RUN_A, "run_status": "active", "poll_after_ms": 100,
        "events": [{"event_id": "e1", "type": "speech", "payload": {}, "run_id": RUN_B}],
        "turn": None,
    })
    assert pr2.events[0].run_id == RUN_B


def test_run_scope_uses_top_level_run_id_for_simple_game_ids():
    """codex PR#27 P2: with simple game ids like 'game_1' (no '<run>_game_<NNN>' prefix), run scope
    must still key by the poll's run_id — carrying within a run and separating different runs that
    reuse the same simple game ids in one harness process."""
    # Two different runs, each with a plain 'game_1' / 'game_2' id scheme.
    a1 = PollResponse.from_dict({"signup_id": "s", "run_id": RUN_A, "run_status": "active",
        "events": [], "turn": {"turn_id": "t", "game_instance_id": "game_1", "game": "onuw",
        "seat": 0, "phase": "p", "action_kind": "k", "deadline_at": "d",
        "observation": {}, "legal_action": {}}, "poll_after_ms": 1}).turn
    a2 = PollResponse.from_dict({"signup_id": "s", "run_id": RUN_A, "run_status": "active",
        "events": [], "turn": {"turn_id": "t", "game_instance_id": "game_2", "game": "onuw",
        "seat": 0, "phase": "p", "action_kind": "k", "deadline_at": "d",
        "observation": {}, "legal_action": {}}, "poll_after_ms": 1}).turn
    b1 = PollResponse.from_dict({"signup_id": "s", "run_id": RUN_B, "run_status": "active",
        "events": [], "turn": {"turn_id": "t", "game_instance_id": "game_1", "game": "onuw",
        "seat": 0, "phase": "p", "action_kind": "k", "deadline_at": "d",
        "observation": {}, "legal_action": {}}, "poll_after_ms": 1}).turn

    assert state_key(a1, reset_between_games=False) == RUN_A
    assert state_key(a2, reset_between_games=False) == RUN_A   # same run -> one key across games
    assert state_key(b1, reset_between_games=False) == RUN_B   # different run -> different key
    assert state_key(a1, reset_between_games=False) != state_key(b1, reset_between_games=False)


def test_state_path_name_is_a_safe_single_segment():
    # Safe: one path segment, no separators, readable sanitized stem preserved.
    name = state_path_name("run_aaa:run_aaa_game_001")
    assert name.startswith("run_aaa_run_aaa_game_001-")
    assert "/" not in state_path_name("a/b\\c d")
    assert "\\" not in state_path_name("a/b\\c d")
    assert state_path_name("///").startswith("state-")     # empty-after-sanitize still resolves


def test_state_path_name_is_injective_for_distinct_keys():
    # codex PR#27 P2: sanitization alone is many-to-one ('run:a' and 'run_a' both -> 'run_a'),
    # which would point two distinct run keys at the same FileMemoryAgent file. The hash suffix
    # must keep distinct keys on distinct paths so run separation is never corrupted.
    assert state_path_name("run:a") != state_path_name("run_a")
    assert state_path_name("run/a") != state_path_name("run_a")
    keys = ["run:a", "run_a", "run/a", "run.a", "run-a", "RUN_A", "run_a "]
    assert len({state_path_name(k) for k in keys}) == len(keys)
    # Deterministic: same key -> same path across calls.
    assert state_path_name("run:a") == state_path_name("run:a")


def test_game_scope_key_does_not_double_run_id():
    """codex PR#27 r2: in the arena shape gid == f'{run_id}_game_NNN'. The default per-game key must
    be the raw gid, NOT run_id-prefixed — re-prefixing doubles an (uncapped) run id into the key."""
    run_id = "my_long_run_id_2026"
    gid = f"{run_id}_game_007"
    t = Turn.from_dict({
        "turn_id": "t", "game_instance_id": gid, "game": "onuw", "seat": 0, "phase": "p",
        "action_kind": "k", "deadline_at": "d", "observation": {}, "legal_action": {},
        "run_id": run_id,
    })
    assert state_key(t, reset_between_games=True) == gid          # raw gid, no "run_id:" prefix
    assert run_id + ":" not in state_key(t, reset_between_games=True)


def test_state_path_name_is_length_bounded_for_long_keys():
    """codex PR#27 r2: run/game ids are not length-capped upstream, so a raw key can blow the
    255-byte filename limit. The path segment must stay well under it for any key."""
    long_key = "r" * 4000 + ":" + "g" * 4000
    name = state_path_name(long_key)
    assert len(name) <= 255
    assert len(name.encode("utf-8")) < 255
    # Still injective at the extreme: two different long keys map to different segments.
    assert state_path_name("x" * 5000) != state_path_name("y" * 5000)


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
        f"state_key={_gid(RUN_A, 1)}; reset_between_games=True; "
        f"events_seen=2; turns_seen=2"
    )

    # Game 2 in the SAME run starts fresh: no events/turns carried over.
    out2 = probe.act(_turn(RUN_A, 2))
    assert "events_seen=0" in out2["reasoning"]
    assert "turns_seen=1" in out2["reasoning"]
    assert f"state_key={_gid(RUN_A, 2)}" in out2["reasoning"]

    # The two games are tracked under distinct (raw game-id) keys.
    assert set(probe.turns) == {_gid(RUN_A, 1), _gid(RUN_A, 2)}


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


# --- Real-run schema validity (codex PR#27 P2): the probe must emit actions the connected ONUW
#     schema ACCEPTS, else `arena-agent play` rejects/forfeits every turn to the deadline. -------

def _probe_replies_drive_a_real_connected_batch(tmp_path, monkeypatch, n_games=2):
    """Run a real connected ONUW batch where every reply comes from StateProbeAgent.act, and return
    (run, invalid_action_kinds). Mirrors test_connected_runner's store-backed responder, but routes
    actions through the probe so the production `_validate_action` (store.reply_to_turn) judges them."""
    import threading
    import time
    from types import SimpleNamespace
    from arena import store
    from arena.connected import run_connected_batch
    from examples.state_probe_agent import StateProbeAgent

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "probe_real.db")
    store.init_schema()
    store.create_connected_run({
        "id": "probe_run", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": n_games, "players": 5, "seed_base": 11,
    })
    agent_by_signup = {}
    for i in range(5):
        a = store.register_agent(f"probe-{i}", f"hash_probe_{i}", "arena-agent-v1", "test")
        s, err = store.create_signup("probe_run", a["id"])
        assert err is None
        agent_by_signup[s["id"]] = a["id"]
    for sid, aid in agent_by_signup.items():
        store.mark_signup_ready(sid, aid)

    probe = StateProbeAgent(reset_between_games=True)
    invalid: list[str] = []
    stop = threading.Event()

    def responder():
        while not stop.is_set():
            for sid, aid in agent_by_signup.items():
                t = store.pending_turn_for_signup(sid)
                if not t:
                    continue
                ns = SimpleNamespace(action_kind=t["action_kind"], legal_action=t["legal_action"],
                                     game_instance_id=t.get("game_instance_id"), run_id=t.get("run_id"))
                out = probe.act(ns)
                _reply, err = store.reply_to_turn(t["id"], aid, out["action"], out.get("reasoning"), 1)
                if err == "invalid_action":
                    invalid.append(t["action_kind"])
            if {s["status"] for s in store.list_run_signups("probe_run")} == {"completed"}:
                return
            time.sleep(0.01)

    th = threading.Thread(target=responder)
    th.start()
    try:
        run_connected_batch("probe_run", discussion_rounds=1)
    finally:
        stop.set()
    th.join(timeout=5)
    return store.get_run("probe_run"), invalid


def test_probe_actions_accepted_by_real_connected_schema(tmp_path, monkeypatch):
    """codex PR#27 P2: every action the probe emits in a real connected ONUW game must be accepted
    by the production schema validator (no 'invalid_action'), and the run completes without forfeits."""
    run, invalid = _probe_replies_drive_a_real_connected_batch(tmp_path, monkeypatch)
    assert invalid == [], f"probe emitted schema-invalid actions for: {sorted(set(invalid))}"
    assert run["status"] == "done", f"run did not complete cleanly: {run['status']}"


def test_event_positional_contract_unchanged_by_run_id():
    """codex PR#27 final: run_id must be the LAST Event field. A legacy positional caller
    Event(event_id, type, payload, seq, game_instance_id) must still bind seq/game_instance_id
    correctly (not shift them into run_id)."""
    e = Event("e1", "speech", {"x": 1}, 7, "run_z_game_002")
    assert e.seq == 7
    assert e.game_instance_id == "run_z_game_002"
    assert e.run_id is None
    # state keying off such an event still works (recovers run_id from the game-id prefix).
    assert state_key(e, reset_between_games=False) == "run_z"
