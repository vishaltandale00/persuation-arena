"""Guards the onboarding tidy: the WolfForge experiment rosters live under
experiments/wolfforge/ and are loadable via ARENA_AGENTS_FILE, while the default
roster stays agents.yaml (the only roster Modal ships).

This pins the relocated paths so a future move/delete can't silently break the
documented `ARENA_AGENTS_FILE=experiments/wolfforge/...` workflow.
"""
from __future__ import annotations

import os

import pytest

from arena.config import ROOT, Settings


def test_wolfforge_files_relocated_and_old_paths_gone():
    assert (ROOT / "experiments" / "wolfforge" / "agents.wolfforge.yaml").is_file()
    assert (ROOT / "experiments" / "wolfforge" / "agents.wolfforge.blind.yaml").is_file()
    # The old repo-root copies must be gone.
    assert not (ROOT / "agents.wolfforge.yaml").exists()
    assert not (ROOT / "agents.wolfforge.blind.yaml").exists()


def test_default_roster_is_agents_yaml(monkeypatch):
    monkeypatch.delenv("ARENA_AGENTS_FILE", raising=False)
    roster = Settings().roster()
    assert len(roster) > 0
    # Modal ships only agents.yaml; sanity-check it loads as the default.
    assert all(spec.model for spec in roster)


@pytest.mark.parametrize(
    "rel_path, expected_names",
    [
        (
            "experiments/wolfforge/agents.wolfforge.yaml",
            {"BayesWolf", "CharmWolf", "ChaosWolf", "Control-A", "Control-B"},
        ),
        (
            "experiments/wolfforge/agents.wolfforge.blind.yaml",
            {"Cedar", "Quartz", "Indigo", "Maple", "Slate"},
        ),
    ],
)
def test_wolfforge_roster_loads_from_new_path(monkeypatch, rel_path, expected_names):
    monkeypatch.setenv("ARENA_AGENTS_FILE", rel_path)
    roster = Settings().roster()
    assert {spec.name for spec in roster} == expected_names
    # Harness is what varies in the experiment; model/decoding are held constant.
    assert {spec.harness for spec in roster} == {"bayes", "charisma", "chaos", "base"}
