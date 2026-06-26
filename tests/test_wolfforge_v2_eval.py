"""WolfForgeAgentV2 evaluation tests: pure statistics + store-backed fixture analysis. No network."""
from __future__ import annotations

from arena import store

from tools.wolfforge_v2_eval import (
    GameObs,
    analyze,
    bootstrap_diff_ci,
    build_report,
    load_observations,
    manifest_warnings,
    mcnemar_exact,
    run_manifest,
)


# --- pure statistics ------------------------------------------------------------------------------

def test_mcnemar_exact_basics():
    assert mcnemar_exact(0, 0) == 1.0          # no discordant pairs
    assert mcnemar_exact(5, 5) == 1.0          # perfectly balanced
    # 10 discordant pairs all favoring V2 -> strong signal
    assert mcnemar_exact(10, 0) < 0.01
    # symmetric in its arguments
    assert mcnemar_exact(8, 2) == mcnemar_exact(2, 8)


def _obs(v2_results, base_results, roles=None, teams=None):
    roles = roles or ["Villager"] * len(v2_results)
    teams = teams or ["good"] * len(v2_results)
    return [
        GameObs(run_id="r", gid=i, v2_won=bool(v2_results[i]), baseline_won=bool(base_results[i]),
                v2_role=roles[i], baseline_role="Villager", v2_team=teams[i], baseline_team="good",
                v2_calls=10, v2_forfeits=0, baseline_calls=10, baseline_forfeits=0)
        for i in range(len(v2_results))
    ]


def test_bootstrap_resamples_whole_games_and_is_seed_stable():
    obs = _obs([1, 1, 1, 0, 1, 1, 0, 1, 1, 1], [0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    lo1, hi1 = bootstrap_diff_ci(obs, iterations=2000, seed=7)
    lo2, hi2 = bootstrap_diff_ci(obs, iterations=2000, seed=7)
    assert (lo1, hi1) == (lo2, hi2)            # deterministic for a fixed seed
    assert lo1 <= hi1
    # V2 clearly ahead here -> interval should be positive-leaning
    assert hi1 > 0


def test_analyze_effect_and_pairing():
    # V2 wins 7, baseline wins 3, over 10 games
    obs = _obs([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    a = analyze(obs, bootstrap_iterations=2000)
    assert a.n_games == 10
    assert a.v2_wins == 7 and a.baseline_wins == 3
    assert round(a.effect_pp, 1) == 40.0
    # discordant: V2 won & base lost (b) vs base won & V2 lost (c)
    assert a.discordant_b == 4 and a.discordant_c == 0
    assert a.v2_forfeit_rate == 0.0


def test_analyze_role_stratification():
    obs = _obs([1, 0, 1, 1], [0, 0, 0, 0], roles=["Werewolf", "Werewolf", "Seer", "Seer"])
    a = analyze(obs, bootstrap_iterations=500)
    assert a.by_role["Werewolf"]["games"] == 2
    assert a.by_role["Seer"]["games"] == 2
    assert a.by_role["Seer"]["v2_wins"] == 2


# --- store-backed fixtures ------------------------------------------------------------------------

def _sqlite_store(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "eval.db")
    store.init_schema()


def _save_run(run_id, *, model="openai/gpt-4o-mini", temperature=0.35, rounds=4, deck="arena",
              n_games=4):
    agents = [
        {"name": "WolfForgeV2", "model": model, "harness": "wolfforge-v2",
         "temperature": temperature, "reasoning_effort": "medium", "max_tokens": 4000},
        {"name": "CharismaBaseline", "model": model, "harness": "charisma-baseline",
         "temperature": temperature, "reasoning_effort": "medium", "max_tokens": 4000},
        {"name": "Opp1", "model": model, "harness": "base", "temperature": temperature,
         "reasoning_effort": "medium", "max_tokens": 4000},
    ]
    store.save_run({
        "id": run_id, "game": "onuw", "label": "ONUW", "status": "done",
        "n_games": n_games, "players": 5, "seed_base": 1000, "created": "2026-06-25 00:00",
        "agents": agents, "deck_preset": deck,
        "metadata": {"run_config": {"discussion_rounds": rounds, "temperature": temperature,
                                     "max_tokens_per_turn": 4000, "reasoning_effort": "medium"}},
    })


def _save_game(run_id, gid, *, v2_won, base_won, v2_role="Villager", v2_team="good"):
    """Persist a minimal 5-seat game where seat0=V2, seat1=CharismaBaseline, rest opponents."""
    players = [
        {"seat": 0, "dealt": v2_role, "end": v2_role, "team": v2_team, "believes": v2_role,
         "won": v2_won, "calls": 10, "forfeits": 0},
        {"seat": 1, "dealt": "Villager", "end": "Villager", "team": "good", "believes": "Villager",
         "won": base_won, "calls": 10, "forfeits": 0},
        {"seat": 2, "dealt": "Werewolf", "end": "Werewolf", "team": "evil", "believes": "Werewolf",
         "won": not v2_won, "calls": 10, "forfeits": 0},
        {"seat": 3, "dealt": "Seer", "end": "Seer", "team": "good", "believes": "Seer",
         "won": v2_won, "calls": 10, "forfeits": 0},
        {"seat": 4, "dealt": "Minion", "end": "Minion", "team": "evil", "believes": "Minion",
         "won": not v2_won, "calls": 10, "forfeits": 0},
    ]
    transcript = {"seed": 1000 + gid, "winner_team": "good" if v2_won else "evil",
                  "outcome": {"team": "good" if v2_won else "evil", "text": "fixture"},
                  "players": players}
    meta = [
        {"name": "WolfForgeV2", "model": "openai/gpt-4o-mini"},
        {"name": "CharismaBaseline", "model": "openai/gpt-4o-mini"},
        {"name": "Opp1", "model": "openai/gpt-4o-mini"},
        {"name": "Opp2", "model": "openai/gpt-4o-mini"},
        {"name": "Opp3", "model": "openai/gpt-4o-mini"},
    ]
    store.save_game(run_id, gid, transcript, meta)


def test_load_and_analyze_fixture_runs(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    _save_run("holdout_a")
    # V2 wins 3 of 4, baseline wins 1 of 4
    _save_game("holdout_a", 1, v2_won=True, base_won=False, v2_role="Werewolf", v2_team="evil")
    _save_game("holdout_a", 2, v2_won=True, base_won=False)
    _save_game("holdout_a", 3, v2_won=False, base_won=True)
    _save_game("holdout_a", 4, v2_won=True, base_won=False)

    obs = load_observations(["holdout_a"], "WolfForgeV2", "CharismaBaseline", store)
    assert len(obs) == 4
    a = analyze(obs, bootstrap_iterations=1000)
    assert a.v2_wins == 3 and a.baseline_wins == 1
    assert a.discordant_b == 3 and a.discordant_c == 1
    assert "Werewolf" in a.by_role


def test_manifest_warns_on_mismatched_config(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    _save_run("run_good", model="openai/gpt-4o-mini", temperature=0.35)
    _save_run("run_bad", model="openai/gpt-4o", temperature=0.7)  # different model + temperature
    for r in ("run_good", "run_bad"):
        _save_game(r, 1, v2_won=True, base_won=False)

    manifests = [run_manifest("run_good", store), run_manifest("run_bad", store)]
    warnings = manifest_warnings(manifests)
    assert any("model" in w for w in warnings)
    assert any("temperature" in w for w in warnings)


def test_frozen_manifest_cross_check(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    _save_run("blk", model="openai/gpt-4o-mini", temperature=0.35)
    _save_game("blk", 1, v2_won=True, base_won=False)
    manifests = [run_manifest("blk", store)]
    # frozen value matches what the run used -> no warning
    assert manifest_warnings(manifests, frozen={"model": "openai/gpt-4o-mini"}) == []
    # frozen value differs -> warning
    warns = manifest_warnings(manifests, frozen={"model": "openai/gpt-4o"})
    assert any("frozen manifest" in w and "model" in w for w in warns)


def test_clean_manifest_has_no_warnings(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    _save_run("blk_a")
    _save_run("blk_b")
    for r in ("blk_a", "blk_b"):
        _save_game(r, 1, v2_won=True, base_won=False)
    manifests = [run_manifest("blk_a", store), run_manifest("blk_b", store)]
    assert manifest_warnings(manifests) == []


def test_build_report_renders_markdown(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    _save_run("rep")
    for gid in range(1, 5):
        _save_game("rep", gid, v2_won=(gid != 3), base_won=(gid == 3))
    report = build_report(["rep"], "WolfForgeV2", "CharismaBaseline", store,
                          bootstrap_iterations=1000)
    md = report["markdown"]
    assert "# WolfForgeAgentV2 evaluation report" in md
    assert "Primary result" in md and "Paired test" in md
    assert "Honest conclusion" in md
    assert report["warnings"] == []
    assert report["analysis"].n_games == 4
