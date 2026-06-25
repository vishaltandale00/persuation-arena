from __future__ import annotations

import sys

import pytest


def test_run_cli_passes_run_level_caps(monkeypatch):
    from arena import batch, cli

    captured = {}

    def fake_run_batch(**kwargs):
        captured.update(kwargs)
        return kwargs["run_id"]

    monkeypatch.setattr(batch, "run_batch", fake_run_batch)
    monkeypatch.setattr(sys, "argv", [
        "arena", "run",
        "--game", "onuw",
        "--games", "1",
        "--seed", "42",
        "--run-id", "r_cli",
        "--workers", "1",
        "--rounds", "3",
        "--deal-schedule", "balanced",
        "--reasoning-effort", "high",
        "--max-tokens-per-turn", "777",
        "--temperature", "0.1",
        "--retries", "2",
        "--prior-message-turns", "0",
    ])

    cli.main()

    caps = captured["caps"]
    assert captured["run_id"] == "r_cli"
    assert captured["discussion_rounds"] == 3
    assert captured["deal_schedule"] == "balanced"
    assert caps.reasoning_effort == "high"
    assert caps.max_tokens_per_turn == 777
    assert caps.temperature == 0.1
    assert caps.retries == 2
    assert caps.prior_message_turns == 0


def test_run_cli_rejects_invalid_caps(monkeypatch):
    from arena import cli

    monkeypatch.setattr(sys, "argv", [
        "arena", "run",
        "--max-tokens-per-turn", "0",
    ])

    with pytest.raises(SystemExit):
        cli.main()
