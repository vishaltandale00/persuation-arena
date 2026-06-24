"""Run-layer tests: fresh-deal schedule, batch resilience, per-role + forfeit scoring,
and agent forfeit telemetry. No API calls (the model client / _play_one are stubbed)."""
from __future__ import annotations

from types import SimpleNamespace

from arena import batch, store
from arena.config import AgentSpec


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


def test_every_scheduled_deal_has_a_wolf_in_play():
    # require_wolf_in_play guarantees at least one Werewolf is dealt (never all benched in the
    # center), so no scheduled game is an unrateable no-wolf round. This seed_base previously led
    # with a zero-wolf deal; the guarantee now precludes it across the whole cycle.
    from arena.games.onuw import ONUW

    counts = []
    for _, seed, _ in batch.fresh_deal_schedule(n_games=5, n_players=5, seed_base=702742):
        core = ONUW({i: f"P{i}" for i in range(5)}, seed=seed)
        core.deal()
        counts.append(list(core.dealt.values()).count("Werewolf"))
    assert all(c >= 1 for c in counts), counts


# ---- batch resilience -------------------------------------------------------
def _fake_play_one(core_cls, specs, n_players, seed, gid, rot, discussion_rounds, deck_preset=None):
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
    assert resp.reasoning == "r"
    assert resp.declared_reasoning == "r"
    assert resp.provider_reasoning == "native trace"
    assert a.calls[0]["declared_reasoning"] == "r"
    assert a.calls[0]["provider_reasoning"] == "native trace"
    assert a.calls[0]["reasoning_effort"] == openrouter.SETTINGS.caps.reasoning_effort
    assert a.calls[0]["max_tokens"] == openrouter.SETTINGS.caps.max_tokens_per_turn
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
