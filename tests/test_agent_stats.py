from __future__ import annotations

import types

from arena.games.base import agent_stats


def test_counts_calls_and_forfeits_with_missing_ok_defaulting_true():
    agent = types.SimpleNamespace(calls=[{"ok": True}, {"ok": False}, {"ok": True}, {}])
    assert agent_stats(agent) == {"calls": 4, "forfeits": 1}


def test_object_without_calls_attr_reports_zeros():
    agent = types.SimpleNamespace(name="scripted")
    assert agent_stats(agent) == {"calls": 0, "forfeits": 0}


def test_calls_none_reports_zeros():
    agent = types.SimpleNamespace(calls=None)
    assert agent_stats(agent) == {"calls": 0, "forfeits": 0}


def test_empty_calls_reports_zeros():
    agent = types.SimpleNamespace(calls=[])
    assert agent_stats(agent) == {"calls": 0, "forfeits": 0}


def test_all_false_calls_are_all_forfeits():
    agent = types.SimpleNamespace(calls=[{"ok": False}, {"ok": False}, {"ok": False}])
    assert agent_stats(agent) == {"calls": 3, "forfeits": 3}


def test_explicit_ok_true_never_counts_as_forfeit():
    agent = types.SimpleNamespace(calls=[{"ok": True, "ms": 12, "raw": "x"}])
    assert agent_stats(agent) == {"calls": 1, "forfeits": 0}
