"""Run-layer tests: fresh-deal schedule, batch resilience, per-role + forfeit scoring,
and agent forfeit telemetry. No API calls (the model client / _play_one are stubbed)."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest
from types import SimpleNamespace

from arena import batch, store
from arena.config import AgentSpec
from arena.games.onuw import default_deck


SPECS = [AgentSpec(name=f"A{i}", model=f"m{i}") for i in range(5)]


# ---- fresh-deal schedule ----------------------------------------------------
def test_schedule_uses_fresh_seed_every_game_and_rotates_seats():
    sched = batch.fresh_deal_schedule(n_games=20, n_players=5, seed_base=9000)
    assert len(sched) == 20
    assert [seed for _, seed, _ in sched] == list(range(9000, 9020))
    # Within each 5-game cycle, every rotation 0..4 appears exactly once.
    for b in range(4):
        block = sched[b * 5:(b + 1) * 5]
        assert sorted(r for _, _, r in block) == [0, 1, 2, 3, 4]
    assert [g for g, _, _ in sched] == list(range(1, 21))


def test_schedule_partial_cycle_continues_fresh_deals():
    sched = batch.fresh_deal_schedule(n_games=7, n_players=5, seed_base=100)
    assert [(g, s, r) for g, s, r in sched] == [
        (1, 100, 0), (2, 101, 1), (3, 102, 2), (4, 103, 3), (5, 104, 4),
        (6, 105, 0), (7, 106, 1),
    ]


def test_schedule_rotation_gives_each_agent_each_seat_once_per_cycle():
    # Over one cycle, agent at spec-index k occupies seats (k+rot)%5 for rot in 0..4 -> all 5 seats.
    sched = batch.fresh_deal_schedule(n_games=5, n_players=5, seed_base=0)
    seats_for_agent = {k: set() for k in range(5)}
    for _, _, rot in sched:
        for seat in range(5):
            spec_idx = (seat + rot) % 5  # seat_to_spec[seat] = specs[(seat+rot)%n]
            seats_for_agent[spec_idx].add(seat)
    assert all(seats == {0, 1, 2, 3, 4} for seats in seats_for_agent.values())


def test_shard_slice_partitions_global_schedule():
    """V-1 (REQ-1): slicing fresh_deal_schedule by (shard_index, num_shards) partitions the
    GLOBAL schedule with no loss, no overlap, and global gid/seed/rot preserved."""
    cases = [
        (20, 5, 9000),  # N divisible by several K
        (7, 5, 100),    # N not divisible by 5/3/4
        (5, 5, 0),      # N == n_players
        (13, 3, 42),    # prime-ish N, small player count
    ]
    for N, n, seed in cases:
        full = batch.fresh_deal_schedule(N, n, seed)
        full_by_gid = {gid: (seed_, rot) for gid, seed_, rot in full}

        for K in {1, 2, 3, 4, N}:
            slices = [
                batch.fresh_deal_schedule(N, n, seed, shard_index=k, num_shards=K)
                for k in range(K)
            ]

            # Every kept entry preserves the GLOBAL (seed, rot) for its gid (no renumbering),
            # and its 0-indexed position g=gid-1 satisfies the stride predicate g % K == k.
            for k, sl in enumerate(slices):
                for gid, seed_, rot in sl:
                    assert (seed_, rot) == full_by_gid[gid], (N, n, seed, K, k, gid)
                    assert (gid - 1) % K == k, (N, n, seed, K, k, gid)

            # Slices pairwise disjoint on gid.
            gid_sets = [{gid for gid, _, _ in sl} for sl in slices]
            for i in range(K):
                for j in range(i + 1, K):
                    assert gid_sets[i].isdisjoint(gid_sets[j]), (N, n, seed, K, i, j)

            # Union of the K slices == full schedule (as a set of (gid, seed, rot)).
            union = {entry for sl in slices for entry in sl}
            assert union == set(full), (N, n, seed, K)

            # Slice sizes differ by at most 1 (stride partition is as even as possible).
            sizes = [len(sl) for sl in slices]
            assert max(sizes) - min(sizes) <= 1, (N, n, seed, K, sizes)

            if K == 1:
                # K=1 slice == full schedule, in order.
                assert slices[0] == full
            if K == N:
                # K=N ⇒ each slice has exactly one game.
                assert all(len(sl) == 1 for sl in slices)


def test_balanced_onuw_schedule_balances_model_role_exposure():
    deck = default_deck(5, "arena")
    sched = batch.balanced_onuw_deal_schedule(n_games=40, n_players=5, seed_base=9000, deck=deck)
    assert len(sched) == 40
    for _, _, _, deal in sched:
        assert deal is not None
        assert Counter(deal) == Counter(deck)
    assert any("Werewolf" not in deal[:5] for _, _, _, deal in sched)

    exposure = batch.schedule_role_exposure(sched, 5, SPECS)
    for role in set(deck):
        counts = [exposure[spec.name].get(role, 0) for spec in SPECS]
        assert max(counts) - min(counts) <= 1


def test_random_deal_allows_all_wolves_in_center():
    from arena.games.onuw import ONUW

    core = ONUW({i: f"P{i}" for i in range(5)}, seed=22)
    core.deal()

    assert list(core.dealt.values()).count("Werewolf") == 0
    assert core.center.count("Werewolf") == 2


# ---- batch resilience -------------------------------------------------------
def _fake_play_one(core_cls, specs, n_players, seed, gid, rot, discussion_rounds, deck_preset=None,
                   caps=None, run_id=None, stream_events=False, deal_override=None):
    if gid == 2:
        raise RuntimeError("boom")
    players = [{"seat": i, "dealt": "Villager", "end": "Villager", "team": "good",
                "believes": "Villager", "won": False, "calls": 3, "forfeits": 0}
               for i in range(n_players)]
    t = {"seed": seed, "winner_team": "good", "outcome": {"text": "ok", "team": "good"},
         "players": players}
    meta = [{"name": specs[(i + rot) % n_players].name, "model": specs[(i + rot) % n_players].model}
            for i in range(n_players)]
    return gid, t, meta


def test_run_batch_survives_one_failing_game(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(batch, "_play_one", _fake_play_one)
    rid = batch.run_batch(game="onuw", n_games=5, seed_base=500, run_id="r1",
                          roster=SPECS, workers=2)
    run = store.get_run(rid)
    assert run["status"] == "partial"                 # one game failed -> not 'done'
    assert store.distinct_gids(rid) == [1, 3, 4, 5]    # gid 2 dropped, rest persisted


def test_run_batch_persists_effective_run_config(tmp_path, monkeypatch):
    from arena.config import caps_with_overrides

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(batch, "_play_one", _fake_play_one)
    caps = caps_with_overrides(
        reasoning_effort="high",
        max_tokens_per_turn=1234,
        temperature=0.2,
        retries=3,
        prior_message_turns=0,
        discussion_rounds=2,
    )

    rid = batch.run_batch(game="onuw", n_games=1, seed_base=500, run_id="r_cfg",
                          roster=SPECS, workers=1, discussion_rounds=2, caps=caps)

    run = store.get_run(rid)
    assert run["metadata"]["run_config"] == {
        "discussion_rounds": 2,
        "reasoning_effort": "high",
        "max_tokens_per_turn": 1234,
        "temperature": 0.2,
        "retries": 3,
        "prior_message_turns": 0,
    }
    assert run["metadata"]["deal_schedule"]["mode"] == "balanced"
    assert run["agents"][0]["reasoning_effort"] == "high"
    assert run["agents"][0]["max_tokens_per_turn"] == 1234
    assert run["agents"][0]["temperature"] == 0.2
    assert run["agents"][0]["retries"] == 3
    assert run["agents"][0]["prior_message_turns"] == 0
    assert run["agents"][0]["discussion_rounds"] == 2


def test_run_batch_skips_existing_and_publishes_new_games(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(batch, "_play_one", _fake_play_one)
    published = []

    rid = batch.run_batch(game="onuw", n_games=3, seed_base=500, run_id="r1",
                          roster=SPECS, workers=1, skip_gids={2},
                          on_game_saved=lambda gid, t, meta: published.append(gid))

    assert rid == "r1"
    assert store.distinct_gids(rid) == [1, 3]
    assert published == [1, 3]


def test_run_batch_persists_balanced_schedule_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(batch, "_play_one", _fake_play_one)

    rid = batch.run_batch(game="onuw", n_games=10, seed_base=500, run_id="r_bal",
                          roster=SPECS, workers=1, deal_schedule="balanced")

    meta = store.get_run(rid)["metadata"]["deal_schedule"]
    assert meta["mode"] == "balanced"
    assert meta["algorithm"] == "role_deficit_v1"
    assert set(meta["role_exposure"]) == {s.name for s in SPECS}
    assert all(sum(roles.values()) == 10 for roles in meta["role_exposure"].values())


def test_run_batch_preserves_explicit_random_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(batch, "_play_one", _fake_play_one)

    rid = batch.run_batch(game="onuw", n_games=1, seed_base=500, run_id="r_random",
                          roster=SPECS, workers=1, deal_schedule="random")

    assert store.get_run(rid)["metadata"]["deal_schedule"] == {"mode": "random"}


# ---- per-role + forfeit scoring --------------------------------------------
def _save_game(rid, gid, rows):
    players = [{"seat": i, "dealt": d, "end": d, "team": team, "believes": d,
                "won": won, "calls": calls, "forfeits": ff}
               for i, (d, team, won, calls, ff) in enumerate(rows)]
    t = {"seed": gid, "winner_team": "good", "outcome": {"text": "t", "team": "good"},
         "players": players}
    meta = [{"name": f"A{i}", "model": f"m{i}"} for i in range(len(rows))]
    store.save_game(rid, gid, t, meta)


def test_score_run_reports_per_role_and_forfeit_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store.save_run({"id": "r2", "game": "onuw", "label": "ONUW", "status": "done",
                    "n_games": 2, "players": 5, "seed_base": 0, "created": "now",
                    "agents": [{"name": f"A{i}", "model": f"m{i}", "harness": "base"} for i in range(5)]})
    # A0: game1 Werewolf win (1 forfeit / 5 calls), game2 Seer loss (0 forfeit / 5 calls)
    _save_game("r2", 1, [("Werewolf", "evil", True, 5, 1), ("Seer", "good", False, 5, 0),
                          ("Villager", "good", False, 5, 0), ("Robber", "good", True, 5, 0),
                          ("Villager", "good", False, 5, 0)])
    _save_game("r2", 2, [("Seer", "good", False, 5, 0), ("Werewolf", "evil", True, 5, 0),
                          ("Villager", "good", True, 5, 0), ("Robber", "good", False, 5, 0),
                          ("Villager", "good", False, 5, 0)])
    from arena.score import score_run
    sc = score_run("r2")
    a0 = sc["A0"]
    # A0 was dealt Werewolf once and Seer once
    assert a0["by_role"]["Werewolf"]["n"] == 1 and a0["by_role"]["Werewolf"]["w"] == 1
    assert a0["by_role"]["Seer"]["n"] == 1 and a0["by_role"]["Seer"]["w"] == 0
    assert a0["overall"]["n"] == 2 and a0["overall"]["w"] == 1
    # 1 forfeit over 10 calls
    assert a0["calls"] == 10 and a0["forfeits"] == 1 and a0["forfeit_rate"] == 0.1


# ---- agent forfeit telemetry ------------------------------------------------
def _client_returning(content, *, provider_reasoning=None):
    create = lambda **k: SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(
                content=content,
                reasoning=provider_reasoning,
                reasoning_details=[{"type": "summary", "text": provider_reasoning}] if provider_reasoning else None,
            ),
        )],
        usage={"prompt_tokens": 1, "completion_tokens": 2},
    )
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _set_structured_output(monkeypatch, openrouter, mode):
    monkeypatch.setattr(
        openrouter.SETTINGS,
        "caps",
        replace(openrouter.SETTINGS.caps, openrouter_structured_output=mode),
    )


def test_agent_records_forfeit_on_api_failure(monkeypatch):
    from arena import openrouter

    def boom():
        raise RuntimeError("net")
    bad = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **k: boom())))
    monkeypatch.setattr(openrouter, "client", lambda: bad)
    a = openrouter.OpenRouterAgent("X", "m")
    resp = a.act("obs", lambda action, raw: action, default_action=7)
    assert resp.ok is False and resp.action == 7
    assert len(a.calls) == 1 and a.calls[0]["ok"] is False


def test_agent_defaults_when_parser_rejects_null_action(monkeypatch):
    from arena import openrouter
    monkeypatch.setattr(openrouter, "client",
                        lambda: _client_returning('{"reasoning":"r","action":null}'))
    a = openrouter.OpenRouterAgent("X", "m")
    resp = a.act("obs", lambda action, raw: action.get("move"), default_action="pass")
    assert resp.ok is False and resp.action == "pass"
    assert len(a.calls) == 1 and a.calls[0]["ok"] is False


def test_agent_records_ok_on_valid_response(monkeypatch):
    from arena import openrouter
    monkeypatch.setattr(openrouter, "client",
                        lambda: _client_returning('{"reasoning":"r","action":3}',
                                                  provider_reasoning="native trace"))
    a = openrouter.OpenRouterAgent("X", "m")
    resp = a.act("obs", lambda action, raw: int(action), default_action=0)
    assert resp.ok is True and resp.action == 3
    assert len(a.calls) == 1 and a.calls[0]["ok"] is True
    assert resp.declared_reasoning == "r"
    assert resp.provider_reasoning == "native trace"
    assert a.calls[0]["declared_reasoning"] == "r"
    assert a.calls[0]["provider_reasoning"] == "native trace"
    assert a.calls[0]["reasoning_effort"] == openrouter.SETTINGS.caps.reasoning_effort
    assert a.calls[0]["max_tokens"] == openrouter.SETTINGS.caps.max_tokens_per_turn
    assert a.calls[0]["temperature"] == openrouter.SETTINGS.caps.temperature
    assert a.calls[0]["retries"] == openrouter.SETTINGS.caps.retries
    assert a.calls[0]["prior_message_turns"] == openrouter.SETTINGS.caps.prior_message_turns
    assert a.calls[0]["finish_reason"] == "stop"
    assert a.calls[0]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2}


def test_agent_requests_provider_reasoning(monkeypatch):
    from arena import openrouter

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content='{"reasoning":"r","action":3}'),
            )],
            usage={},
        )

    monkeypatch.setattr(openrouter, "client",
                        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    a = openrouter.OpenRouterAgent("X", "m")
    a.act("obs", lambda action, raw: int(action), default_action=0)

    assert captured["extra_body"]["reasoning"] == {
        "effort": openrouter.SETTINGS.caps.reasoning_effort,
        "exclude": False,
    }
    assert captured["max_tokens"] == openrouter.SETTINGS.caps.max_tokens_per_turn


def test_agent_does_not_request_structured_output_when_disabled(monkeypatch):
    from arena import openrouter

    _set_structured_output(monkeypatch, openrouter, "off")
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content='{"reasoning":"r","action":{"target":2}}'),
            )],
            usage={},
        )

    monkeypatch.setattr(openrouter, "client",
                        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    a = openrouter.OpenRouterAgent("X", "openai/gpt-4o")
    resp = a.act(
        "obs",
        lambda action, raw: int(action["target"]),
        default_action=0,
        action_kind="onuw.vote",
        legal_action={"schema": {
            "type": "object",
            "required": ["target"],
            "properties": {"target": {"type": "integer", "enum": [1, 2]}},
            "additionalProperties": False,
        }},
    )

    assert resp.ok is True and resp.action == 2
    assert "response_format" not in captured
    assert a.calls[0]["structured_output"] == {
        "configured": "off",
        "requested": None,
        "used": None,
        "fallback": False,
    }


def test_agent_requests_json_schema_structured_output_when_enabled(monkeypatch):
    from arena import openrouter

    _set_structured_output(monkeypatch, openrouter, "json_schema")
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content='{"reasoning":"r","action":{"target":2}}'),
            )],
            usage={},
        )

    action_schema = {
        "type": "object",
        "required": ["target"],
        "properties": {"target": {"type": "integer", "enum": [1, 2]}},
        "additionalProperties": False,
    }
    monkeypatch.setattr(openrouter, "client",
                        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    a = openrouter.OpenRouterAgent("X", "openai/gpt-4o")
    resp = a.act(
        "obs",
        lambda action, raw: int(action["target"]),
        default_action=0,
        action_kind="onuw.vote",
        legal_action={"schema": action_schema},
    )

    assert resp.ok is True and resp.action == 2
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "onuw_vote",
            "strict": True,
            "schema": {
                "type": "object",
                "required": ["reasoning", "action"],
                "properties": {
                    "reasoning": {"type": "string"},
                    "action": action_schema,
                },
                "additionalProperties": False,
            },
        },
    }
    assert a.calls[0]["structured_output"] == {
        "configured": "json_schema",
        "requested": "json_schema",
        "used": "json_schema",
        "fallback": False,
    }


def test_agent_falls_back_when_structured_output_is_rejected(monkeypatch):
    from arena import openrouter

    _set_structured_output(monkeypatch, openrouter, "json_schema")
    response_formats = []

    def create(**kwargs):
        response_formats.append(kwargs.get("response_format"))
        if kwargs.get("response_format") is not None:
            raise ValueError("response_format json_schema is not supported by this provider")
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content='{"reasoning":"legacy","action":{"target":1}}'),
            )],
            usage={},
        )

    monkeypatch.setattr(openrouter, "client",
                        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    a = openrouter.OpenRouterAgent("X", "unknown/model")
    resp = a.act(
        "obs",
        lambda action, raw: int(action["target"]),
        default_action=0,
        action_kind="onuw.vote",
        legal_action={"schema": {
            "type": "object",
            "required": ["target"],
            "properties": {"target": {"type": "integer", "enum": [1, 2]}},
            "additionalProperties": False,
        }},
    )

    assert resp.ok is True and resp.action == 1
    assert response_formats[0]["type"] == "json_schema"
    assert response_formats[1] is None
    assert "structured output rejected" in a.calls[0]["validation_error"]
    assert a.calls[0]["structured_output"] == {
        "configured": "json_schema",
        "requested": "json_schema",
        "used": None,
        "fallback": True,
    }


def test_agent_uses_json_object_when_schema_is_unavailable(monkeypatch):
    from arena import openrouter

    _set_structured_output(monkeypatch, openrouter, "json_schema")
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content='{"reasoning":"r","action":"approve"}'),
            )],
            usage={},
        )

    monkeypatch.setattr(openrouter, "client",
                        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    a = openrouter.OpenRouterAgent("X", "openai/gpt-4o")
    resp = a.act("obs", lambda action, raw: str(action), default_action="reject")

    assert resp.ok is True and resp.action == "approve"
    assert captured["response_format"] == {"type": "json_object"}


def test_agent_passes_prior_provider_reasoning_as_assistant_message_fields(monkeypatch):
    from arena import openrouter

    messages_by_call = []
    responses = [
        '{"reasoning":"keep pressure on seat 2","action":3}',
        '{"reasoning":"continue the plan","action":4}',
    ]

    def create(**kwargs):
        messages_by_call.append(kwargs["messages"])
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content=responses.pop(0),
                    reasoning="provider plan",
                    reasoning_details=[{"type": "reasoning.summary", "summary": "provider plan"}],
                ),
            )],
            usage={},
        )

    monkeypatch.setattr(openrouter, "client",
                        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    a = openrouter.OpenRouterAgent("X", "m")
    a.act("first obs", lambda action, raw: int(action), default_action=0, action_kind="vote")
    a.act("second obs", lambda action, raw: int(action), default_action=0, action_kind="vote")

    assert messages_by_call[0] == [
        {"role": "system", "content": openrouter.SYSTEM},
        {"role": "user", "content": "first obs"},
    ]
    assert messages_by_call[1][1] == {"role": "user", "content": "first obs"}
    assert messages_by_call[1][2]["role"] == "assistant"
    assert messages_by_call[1][2]["content"] == '{"reasoning":"keep pressure on seat 2","action":3}'
    assert messages_by_call[1][2]["reasoning_details"] == [
        {"type": "reasoning.summary", "summary": "provider plan"}
    ]
    assert messages_by_call[1][3] == {"role": "user", "content": "second obs"}


def test_agent_response_requires_explicit_declared_reasoning():
    from arena.openrouter import AgentResponse

    with pytest.raises(TypeError):
        AgentResponse(reasoning="ambiguous", action=1, raw="{}", ok=True)

    resp = AgentResponse(
        declared_reasoning="arena rationale",
        action=1,
        raw="{}",
        ok=True,
        provider_reasoning="provider trace",
        provider_reasoning_details=[{"type": "summary", "text": "provider trace"}],
    )
    assert resp.declared_reasoning == "arena rationale"
    assert not hasattr(resp, "reasoning")
    assert resp.provider_reasoning == "provider trace"
    assert resp.provider_reasoning_details == [{"type": "summary", "text": "provider trace"}]
