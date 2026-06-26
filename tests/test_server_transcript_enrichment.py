from arena import server


def test_usage_cost_accepts_agent_call_log_and_legacy_call_log():
    current = {
        "agentCallLog": {
            "0": [
                {"usage": {"cost": 0.1, "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
                {"usage": {"cost": 0.25, "prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}},
            ],
        },
    }
    legacy = {
        "callLog": {
            "1": [
                {"usage": {"cost": 0.05, "prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}},
            ],
        },
    }

    assert server._usage_cost_for_transcript(current) == {
        "cost": 0.35,
        "calls": 2,
        "prompt_tokens": 30,
        "completion_tokens": 5,
        "total_tokens": 35,
    }
    assert server._usage_cost_for_transcript(legacy)["cost"] == 0.05


def test_transcript_enrichment_matches_visible_discussion_actions(monkeypatch):
    transcript = {
        "phases": [
            {
                "name": "Discussion",
                "events": [
                    {"t": "say", "pid": 1, "text": "I need to speak."},
                    {"t": "pass", "pid": 1, "ms": 12},
                ],
            }
        ]
    }
    completed = [
        {
            "game_instance_id": "run_x_game_001",
            "type": "model_turn_completed",
            "phase": "discussion",
            "payload": {
                "seat": 1,
                "action": {"speak": "I need to speak.", "urgency": 3},
                "reasoning": "speak reason",
                "raw": '{"action":{"speak":"I need to speak.","urgency":3}}',
            },
        },
        {
            "game_instance_id": "run_x_game_001",
            "type": "model_turn_completed",
            "phase": "discussion",
            "payload": {
                "seat": 1,
                "action": {"pass": True},
                "reasoning": "pass reason",
                "raw": '{"action":{"pass":true}}',
            },
        },
    ]
    monkeypatch.setattr(server.store, "list_run_events", lambda run_id, max_events=1000: completed)

    enriched = server._enrich_transcript_turn_reasoning("run_x", 1, transcript)
    say, passed = enriched["phases"][0]["events"]

    assert say["declared_reasoning"] == "speak reason"
    assert '"speak"' in say["raw_model_output"]
    assert passed["declared_reasoning"] == "pass reason"
    assert '"pass"' in passed["raw_model_output"]


def test_transcript_enrichment_discards_hidden_passes_before_visible_turns(monkeypatch):
    transcript = {
        "phases": [
            {
                "name": "Discussion",
                "events": [
                    {"t": "say", "pid": 1, "text": "Visible rebuttal."},
                    {"t": "pass", "pid": 1, "ms": 12, "stance": "done"},
                ],
            }
        ]
    }
    completed = [
        {
            "game_instance_id": "run_x_game_001",
            "type": "model_turn_completed",
            "phase": "discussion",
            "payload": {
                "seat": 1,
                "action": {"pass": True, "stance": "wait"},
                "reasoning": "hidden wait reason",
                "raw": '{"action":{"pass":true,"stance":"wait"}}',
            },
        },
        {
            "game_instance_id": "run_x_game_001",
            "type": "model_turn_completed",
            "phase": "discussion",
            "payload": {
                "seat": 1,
                "action": {"speak": "Visible rebuttal.", "urgency": 3},
                "reasoning": "visible speak reason",
                "raw": '{"action":{"speak":"Visible rebuttal.","urgency":3}}',
            },
        },
        {
            "game_instance_id": "run_x_game_001",
            "type": "model_turn_completed",
            "phase": "discussion",
            "payload": {
                "seat": 1,
                "action": {"pass": True, "stance": "done"},
                "reasoning": "visible done reason",
                "raw": '{"action":{"pass":true,"stance":"done"}}',
            },
        },
    ]
    monkeypatch.setattr(server.store, "list_run_events", lambda run_id, max_events=1000: completed)

    enriched = server._enrich_transcript_turn_reasoning("run_x", 1, transcript)
    say, passed = enriched["phases"][0]["events"]

    assert say["declared_reasoning"] == "visible speak reason"
    assert say["hidden_model_turns_before"][0]["action"] == {"pass": True, "stance": "wait"}
    assert say["hidden_model_turns_before"][0]["reasoning"] == "hidden wait reason"
    assert passed["declared_reasoning"] == "visible done reason"
    assert '"stance":"done"' in passed["raw_model_output"]


def test_transcript_enrichment_pass_stance_must_match_visible_event(monkeypatch):
    transcript = {
        "phases": [
            {
                "name": "Discussion",
                "events": [
                    {"t": "pass", "pid": 1, "ms": 12, "stance": "done"},
                ],
            }
        ]
    }
    completed = [
        {
            "game_instance_id": "run_x_game_001",
            "type": "model_turn_completed",
            "phase": "discussion",
            "payload": {
                "seat": 1,
                "action": {"pass": True, "stance": "wait"},
                "reasoning": "hidden wait reason",
                "raw": '{"action":{"pass":true,"stance":"wait"}}',
            },
        },
        {
            "game_instance_id": "run_x_game_001",
            "type": "model_turn_completed",
            "phase": "discussion",
            "payload": {
                "seat": 1,
                "action": {"pass": True, "stance": "done"},
                "reasoning": "visible done reason",
                "raw": '{"action":{"pass":true,"stance":"done"}}',
            },
        },
    ]
    monkeypatch.setattr(server.store, "list_run_events", lambda run_id, max_events=1000: completed)

    enriched = server._enrich_transcript_turn_reasoning("run_x", 1, transcript)
    passed = enriched["phases"][0]["events"][0]

    assert passed["declared_reasoning"] == "visible done reason"
    assert passed["hidden_model_turns_before"][0]["action"] == {"pass": True, "stance": "wait"}
    assert '"stance":"done"' in passed["raw_model_output"]


def test_transcript_enrichment_skips_hidden_losing_bids(monkeypatch):
    transcript = {
        "phases": [
            {
                "name": "Discussion",
                "events": [
                    {"t": "pass", "pid": 1, "ms": 12},
                ],
            }
        ]
    }
    completed = [
        {
            "game_instance_id": "run_x_game_001",
            "type": "model_turn_completed",
            "phase": "discussion",
            "payload": {
                "seat": 1,
                "action": {"speak": "Hidden losing bid.", "urgency": 1},
                "reasoning": "hidden speak reason",
                "raw": '{"action":{"speak":"Hidden losing bid.","urgency":1}}',
            },
        },
        {
            "game_instance_id": "run_x_game_001",
            "type": "model_turn_completed",
            "phase": "discussion",
            "payload": {
                "seat": 1,
                "action": {"pass": True},
                "reasoning": "actual pass reason",
                "raw": '{"action":{"pass":true}}',
            },
        },
    ]
    monkeypatch.setattr(server.store, "list_run_events", lambda run_id, max_events=1000: completed)

    enriched = server._enrich_transcript_turn_reasoning("run_x", 1, transcript)
    passed = enriched["phases"][0]["events"][0]

    assert passed["declared_reasoning"] == "actual pass reason"
    assert '"pass"' in passed["raw_model_output"]


def test_transcript_enrichment_surfaces_defaulted_vote_errors_from_agent_call_log(monkeypatch):
    transcript = {
        "agentCallLog": {
            "1": [
                {
                    "ok": False,
                    "ms": 12.5,
                    "raw": "<error: APIStatusError>",
                    "action_kind": "onuw.vote",
                    "validation_error": "APIStatusError: prompt tokens limit exceeded",
                    "structured_output": {"configured": "off"},
                    "finish_reason": None,
                },
            ],
        },
        "phases": [
            {
                "name": "Vote",
                "events": [
                    {"t": "vote", "pid": 1, "tgt": 3, "ms": 12.5},
                ],
            }
        ],
    }
    completed = [
        {
            "game_instance_id": "run_x_game_001",
            "type": "model_turn_completed",
            "phase": "vote",
            "payload": {
                "seat": 1,
                "action_kind": "onuw.vote",
                "ok": False,
                "ms": 12.5,
                "action": 3,
                "reasoning": "",
                "raw": "<error: APIStatusError>",
            },
        },
    ]
    monkeypatch.setattr(server.store, "list_run_events", lambda run_id, max_events=1000: completed)

    enriched = server._enrich_transcript_turn_reasoning("run_x", 1, transcript)
    vote = enriched["phases"][0]["events"][0]

    assert vote["model_call_ok"] is False
    assert vote["defaulted"] is True
    assert vote["validation_error"] == "APIStatusError: prompt tokens limit exceeded"
    assert vote["raw_model_output"] == "<error: APIStatusError>"
    assert vote["action_kind"] == "onuw.vote"
    assert vote["structured_output"] == {"configured": "off"}


def test_transcript_enrichment_surfaces_event_validation_errors_without_call_log(monkeypatch):
    transcript = {
        "phases": [
            {
                "name": "Vote",
                "events": [
                    {"t": "vote", "pid": 2, "tgt": 0, "ms": 4},
                ],
            }
        ],
    }
    completed = [
        {
            "game_instance_id": "run_x_game_001",
            "type": "model_turn_completed",
            "phase": "vote",
            "payload": {
                "seat": 2,
                "action_kind": "onuw.vote",
                "ok": False,
                "ms": 4,
                "action": 0,
                "validation_error": "ValueError: bad target",
            },
        },
    ]
    monkeypatch.setattr(server.store, "list_run_events", lambda run_id, max_events=1000: completed)

    enriched = server._enrich_transcript_turn_reasoning("run_x", 1, transcript)
    vote = enriched["phases"][0]["events"][0]

    assert vote["model_call_ok"] is False
    assert vote["defaulted"] is True
    assert vote["validation_error"] == "ValueError: bad target"
