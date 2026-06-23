from __future__ import annotations

import threading
import time

from arena import store
from arena.connected import ConnectedAgent, run_connected_batch
from arena.games.onuw import ONUW


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
    assert result["resp"].reasoning == "test"
    assert agent.calls[-1]["ok"] is True


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
            return type("Resp", (), {"action": -1, "reasoning": "vote", "ms": 0.0})()

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
        "n_games": 1, "players": 5, "seed_base": 11,
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
            return {"target": -1}
        if kind == "onuw.seer.inspect":
            return {"mode": "center", "indices": [0, 1]}
        if kind == "onuw.troublemaker.swap_two_or_decline":
            return {"a": None, "b": None}
        if kind in {"onuw.doppelganger.copy_player", "onuw.robber.swap_or_decline"}:
            return {"target": players[0]["seat"] if players else None}
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
    assert len(run["games"]) == 1
    assert {s["status"] for s in store.list_run_signups("run_batch")} == {"completed"}
    event_types = [e["type"] for e in store.list_run_events("run_batch")]
    assert "private_observation" in event_types
    assert "pass" in event_types
    assert "vote_revealed" in event_types
    assert "run_completed" in event_types
