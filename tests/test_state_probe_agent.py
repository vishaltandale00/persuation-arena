from __future__ import annotations

import json

from examples.state_probe_agent import StateProbeAgent
from persuasion_arena_agent.models import Event, Turn


def _event(gid: str, role: str = "Seer") -> Event:
    return Event.from_dict({
        "event_id": f"evt_{gid}",
        "run_id": "run_probe",
        "game_instance_id": gid,
        "type": "role_info",
        "payload": {"seat": 0, "role": role},
    })


def _turn(gid: str) -> Turn:
    return Turn.from_dict({
        "turn_id": f"turn_{gid}",
        "run_id": "run_probe",
        "game_instance_id": gid,
        "game": "onuw",
        "seat": 0,
        "phase": "discussion",
        "action_kind": "onuw.discussion.speak_or_pass",
        "deadline_at": "2026-06-23T17:00:00Z",
        "observation": {"format": "text", "text": "speak"},
        "legal_action": {"schema": {}, "choices": {"players": [{"seat": 1}]}},
    })


def test_state_probe_resets_between_games_by_default(tmp_path):
    log = tmp_path / "probe.jsonl"
    agent = StateProbeAgent(log_path=str(log))

    agent.on_event(_event("run_probe_game_001", "Seer"))
    out1 = agent.act(_turn("run_probe_game_001"))
    agent.on_event(_event("run_probe_game_002", "Robber"))
    out2 = agent.act(_turn("run_probe_game_002"))

    assert "state_key=run_probe:run_probe_game_001" in out1["reasoning"]
    assert "state_key=run_probe:run_probe_game_002" in out2["reasoning"]
    assert "events_seen=1" in out2["reasoning"]
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert {row["key"] for row in rows} == {
        "run_probe:run_probe_game_001",
        "run_probe:run_probe_game_002",
    }


def test_state_probe_can_carry_state_across_run(tmp_path):
    log = tmp_path / "probe.jsonl"
    agent = StateProbeAgent(reset_between_games=False, log_path=str(log))

    agent.on_event(_event("run_probe_game_001", "Seer"))
    out1 = agent.act(_turn("run_probe_game_001"))
    agent.on_event(_event("run_probe_game_002", "Robber"))
    out2 = agent.act(_turn("run_probe_game_002"))

    assert "state_key=run_probe" in out1["reasoning"]
    assert "state_key=run_probe" in out2["reasoning"]
    assert "events_seen=2" in out2["reasoning"]
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert {row["key"] for row in rows} == {"run_probe"}
