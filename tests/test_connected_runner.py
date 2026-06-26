from __future__ import annotations

import threading
import time

from arena import store
from arena.connected import ConnectedAgent, run_connected_batch
from arena.games.onuw import ONUW
from arena.identity import NO_ONE_REF


def _sqlite_store(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "connected.db")
    store.init_schema()


def _active_signup(run_id: str, agent_name: str = "agent") -> tuple[str, str]:
    agent = store.register_agent(agent_name, f"hash_{agent_name}", "arena-agent-v1", "test")
    signup, err = store.create_signup(run_id, agent["id"])
    assert err is None
    ready, err = store.mark_signup_ready(signup["id"], agent["id"])
    assert err is None or err == "not_ready_required"
    return agent["id"], ready["id"]


def test_reply_validation_rejects_invalid_before_first_writer_wins(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "run_validate", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 1, "players": 1, "seed_base": 1,
    })
    agent_id, signup_id = _active_signup("run_validate")
    turn = store.create_turn(
        "run_validate", signup_id, "game_1", 0, "vote", "onuw.vote",
        {"format": "text", "text": "vote"},
        {"schema": {"type": "object", "required": ["target"],
                    "properties": {"target": {"type": "integer", "enum": [1]}},
                    "additionalProperties": False}},
    )

    reply, err = store.reply_to_turn(turn["id"], agent_id, {"target": 2}, "bad")
    assert err == "invalid_action"
    assert store.get_turn_reply(turn["id"]) is None

    reply, err = store.reply_to_turn(turn["id"], agent_id, {"target": 1}, "good")
    assert err is None
    assert reply["accepted"] == 1
    assert store.get_turn_reply(turn["id"])["action"] == {"target": 1}


def test_connected_agent_creates_turn_and_waits_for_reply(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "run_agent", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 1, "players": 1, "seed_base": 1,
    })
    agent_id, signup_id = _active_signup("run_agent")
    agent = ConnectedAgent("agent", "run_agent", signup_id, 0, "game_1", deadline_seconds=2,
                           poll_interval_s=0.02)
    result = {}

    def parse(action, raw):
        return action["speak"]

    def run_act():
        result["resp"] = agent.act(
            "say something",
            parse,
            default_action="pass",
            phase="discussion",
            action_kind="onuw.discussion.speak_or_pass",
            legal_action={"schema": {"type": "object", "required": ["speak"],
                                     "properties": {"speak": {"type": "string"}},
                                     "additionalProperties": False}},
            default_wire_action={"pass": True},
        )

    thread = threading.Thread(target=run_act)
    thread.start()
    deadline = time.time() + 1
    turn = None
    while time.time() < deadline:
        turn = store.pending_turn_for_signup(signup_id)
        if turn:
            break
        time.sleep(0.01)
    assert turn is not None

    reply, err = store.reply_to_turn(turn["id"], agent_id, {"speak": "hello"}, "test", 7)
    assert err is None
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert result["resp"].action == "hello"
    assert result["resp"].declared_reasoning == "test"
    assert agent.calls[-1]["ok"] is True


# --- REQ-7 / V-7: deterministic explicit seats (additive; normal arrival-order untouched) -------


def _signup_with_seat(run_id: str, agent_name: str, seat: int | None):
    """Register a fresh agent and sign it up for run_id, optionally with an explicit seat."""
    agent = store.register_agent(agent_name, f"hash_{agent_name}", "arena-agent-v1", "test")
    signup, err = store.create_signup(run_id, agent["id"], seat=seat)
    assert err is None, err
    return agent["id"], signup["id"]


def test_explicit_seat_honored(tmp_path, monkeypatch):
    """REQ-7 / V-7: create_signup(..., seat=i) -> after the run fills, that signup ends at seat i,
    REGARDLESS of arrival order. Sign up in a deliberately scrambled order vs the requested seats."""
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "run_seats", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 1, "players": 5, "seed_base": 1,
    })
    # arrival order (created_utc): names a,b,c,d,e ; requested seats scramble that order
    requested = [("a", 4), ("b", 2), ("c", 0), ("d", 3), ("e", 1)]
    signup_by_name = {}
    for name, seat in requested:
        _, signup_id = _signup_with_seat("run_seats", name, seat)
        signup_by_name[name] = signup_id

    rows = {s["display_name"]: s for s in store.list_run_signups("run_seats")}
    for name, seat in requested:
        assert rows[name]["seat"] == seat, f"{name} expected seat {seat}, got {rows[name]['seat']}"


def test_identity_to_seat_identical_across_children(tmp_path, monkeypatch):
    """REQ-7 / V-7: two child runs given the SAME roster indices yield the SAME identity->seat map
    (this is what lets sharded children replay identical seats — supports INV-1/V-4)."""
    _sqlite_store(tmp_path, monkeypatch)
    roster = [("Alice", 2), ("Bob", 0), ("Cara", 1)]  # display_name -> requested seat

    def seat_map_for(run_id: str) -> dict[str, int]:
        store.create_connected_run({
            "id": run_id, "game": "onuw", "label": "ONUW", "status": "open",
            "n_games": 1, "players": 3, "seed_base": 1,
        })
        # sign up in a different physical order per child to prove arrival order is NOT what matters
        order = roster if run_id.endswith("0") else list(reversed(roster))
        for name, seat in order:
            agent = store.get_agent_by_token_hash(f"hash_{name}") or store.register_agent(
                name, f"hash_{name}", "arena-agent-v1", "test")
            store.create_signup(run_id, agent["id"], seat=seat)
        return {s["display_name"]: s["seat"] for s in store.list_run_signups(run_id)}

    map0 = seat_map_for("child_0")
    map1 = seat_map_for("child_1")
    assert map0 == map1 == {"Alice": 2, "Bob": 0, "Cara": 1}


def test_arrival_order_unchanged(tmp_path, monkeypatch):
    """INV-2 / V-7: with NO explicit seat, seats are assigned by created_utc arrival order exactly as
    today — the agent signing up first gets seat 0, etc."""
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "run_arrival", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 1, "players": 4, "seed_base": 1,
    })
    arrival = ["first", "second", "third", "fourth"]
    for name in arrival:
        _signup_with_seat("run_arrival", name, None)

    rows = {s["display_name"]: s for s in store.list_run_signups("run_arrival")}
    for i, name in enumerate(arrival):
        assert rows[name]["seat"] == i, f"{name} expected arrival seat {i}, got {rows[name]['seat']}"


def test_onuw_vote_prepares_all_connected_votes_before_waiting():
    names = {i: f"P{i}" for i in range(5)}
    core = ONUW(names, seed=1)
    core.dealt = {i: "Villager" for i in range(5)}
    core.current = dict(core.dealt)
    core.believes = dict(core.dealt)
    core.center = ["Werewolf", "Werewolf", "Minion"]
    prepared: list[int] = []
    waited: list[int] = []

    class Req:
        def __init__(self, seat: int):
            self.seat = seat

        def wait(self):
            assert len(prepared) == 5
            waited.append(self.seat)
            return type("Resp", (), {"action": -1, "declared_reasoning": "vote", "ms": 0.0})()

    class Agent:
        model = "connected-agent"

        def __init__(self, seat: int):
            self.name = f"P{seat}"
            self.seat = seat

        def prepare_act(self, observation, parse_action, default_action, **turn_meta):
            assert turn_meta["action_kind"] == "onuw.vote"
            prepared.append(self.seat)
            return Req(self.seat)

    phase = core.run_vote({i: Agent(i) for i in range(5)})
    assert prepared == [0, 1, 2, 3, 4]
    assert waited == [0, 1, 2, 3, 4]
    assert len(phase["votes"]) == 5


def test_connected_batch_completes_with_store_backed_replies(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "run_batch", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 3, "players": 5, "seed_base": 11,
    })
    agent_by_signup = {}
    for i in range(5):
        agent_id, signup_id = _active_signup("run_batch", f"agent-{i}")
        agent_by_signup[signup_id] = agent_id
    for signup_id, agent_id in agent_by_signup.items():
        ready, err = store.mark_signup_ready(signup_id, agent_id)
        assert err is None or err == "not_ready_required"
    assert {s["status"] for s in store.list_run_signups("run_batch")} == {"active"}

    stop = threading.Event()

    def action_for(turn: dict):
        kind = turn["action_kind"]
        legal = turn["legal_action"]
        players = legal.get("choices", {}).get("players") or []
        if kind == "onuw.discussion.speak_or_pass":
            return {"pass": True}
        if kind == "onuw.vote":
            return {"target": NO_ONE_REF}
        if kind == "onuw.seer.inspect":
            return {"mode": "center", "indices": [0, 1]}
        if kind == "onuw.troublemaker.swap_two_or_decline":
            return {"a": None, "b": None}
        if kind in {"onuw.doppelganger.copy_player", "onuw.robber.swap_or_decline"}:
            return {"target": players[0].get("ref", players[0].get("seat")) if players else None}
        if kind == "onuw.drunk.swap_center":
            return {"index": 0}
        raise AssertionError(kind)

    def responder():
        while not stop.is_set():
            for signup_id, agent_id in agent_by_signup.items():
                turn = store.pending_turn_for_signup(signup_id)
                if not turn:
                    continue
                store.reply_to_turn(turn["id"], agent_id, action_for(turn), "scripted", 1)
            statuses = {s["status"] for s in store.list_run_signups("run_batch")}
            if statuses == {"completed"}:
                return
            time.sleep(0.01)

    thread = threading.Thread(target=responder)
    thread.start()
    try:
        run_connected_batch("run_batch", discussion_rounds=1)
    finally:
        stop.set()
    thread.join(timeout=2)

    run = store.get_run("run_batch")
    assert run["status"] == "done"
    assert len(run["games"]) == 3
    assert {s["status"] for s in store.list_run_signups("run_batch")} == {"completed"}
    events = store.list_run_events("run_batch", max_events=1000)
    event_types = [e["type"] for e in events]
    assert "private_observation" in event_types
    assert "pass" in event_types
    assert "vote_revealed" in event_types
    assert "run_completed" in event_types

    by_game = {}
    for event in events:
        by_game.setdefault(event["game_instance_id"], []).append(event)
    for gid in range(1, 4):
        game_events = by_game[f"run_batch_game_{gid:03d}"]
        vote_observations = [
            event for event in game_events
            if event["phase"] == "vote"
            and event["type"] == "private_observation"
            and event["payload"]["action_kind"] == "onuw.vote"
        ]
        vote_results = [
            event for event in game_events
            if event["phase"] == "vote" and event["type"] == "action_result"
        ]
        vote_reveals = [
            event for event in game_events
            if event["phase"] == "vote" and event["type"] == "vote_revealed"
        ]
        assert len(vote_observations) == 5
        assert len(vote_results) == 5
        assert len(vote_reveals) == 5
