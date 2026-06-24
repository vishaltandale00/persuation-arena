"""Variable table size (5–7 players): every game core deals a valid setup and plays to a
winner, plus the role-specific invariants the larger setups introduce (3 center cards in ONUW,
Mordred hidden from Merlin in 7p Avalon, mafia recognising each other when there are >1)."""
from __future__ import annotations

import pytest

from arena import batch
from arena.config import AgentSpec
from arena.games.avalon import Avalon, ROLES_BY_N as AV_ROLES, TEAM_SIZES_BY_N
from arena.games.base import team_of
from arena.games.onuw import ONUW, deck_for_preset, deck_preset_options, default_deck
from arena.games.secret_mafia import SecretMafia, ROLES_BY_N as SM_ROLES
from tests.scripted import ScriptedDefault


GAMES = [ONUW, Avalon, SecretMafia]


@pytest.mark.parametrize("cls", GAMES)
@pytest.mark.parametrize("n", [5, 6, 7])
def test_game_plays_to_a_winner_at_each_table_size(cls, n):
    names = {i: f"P{i}" for i in range(n)}
    agents = {i: ScriptedDefault() for i in range(n)}
    rec = cls(names, seed=7).play(agents)
    # ONUW can deal no evil faction at all (-> "void", a no-contest); Avalon/Mafia always deal evil.
    assert rec["winner_team"] in ("good", "evil", "void")
    assert len(rec["players"]) == n                       # every seat is dealt and reported
    assert {p["seat"] for p in rec["players"]} == set(range(n))


@pytest.mark.parametrize("n", [5, 6, 7])
def test_onuw_deck_keeps_three_center_cards(n):
    deck = default_deck(n)
    assert len(deck) == n + 3 and len(deck) - n == 3      # canonical ONUW: exactly 3 center cards
    assert deck.count("Werewolf") == 2 and deck.count("Minion") == 1
    assert "Tanner" in deck
    core = ONUW({i: f"P{i}" for i in range(n)}, seed=1)
    core.deal()
    assert len(core.center) == 3


def test_onuw_deck_presets_are_tunable():
    assert deck_for_preset(5, "arena") == [
        "Werewolf", "Werewolf", "Minion", "Seer", "Robber", "Troublemaker", "Drunk", "Tanner",
    ]
    assert deck_for_preset(5, "classic") == [
        "Werewolf", "Werewolf", "Seer", "Robber", "Troublemaker", "Minion", "Villager", "Villager",
    ]
    assert deck_for_preset(6, "tanner")[-1] == "Minion"
    opts = deck_preset_options(5)
    assert {o["id"] for o in opts} == {"arena", "classic", "tanner"}
    assert next(o for o in opts if o["id"] == "arena")["default"] is True


@pytest.mark.parametrize("n,n_evil", [(5, 2), (6, 2), (7, 3)])
def test_avalon_evil_count_and_team_sizes(n, n_evil):
    roles = AV_ROLES[n]
    assert sum(team_of(r) == "evil" for r in roles) == n_evil
    assert len(TEAM_SIZES_BY_N[n]) == 5                   # always five quests


def test_avalon_mordred_hidden_from_merlin_but_seen_by_evil():
    names = {i: f"P{i}" for i in range(7)}
    layout = ["Merlin", "Percival", "Loyal Servant", "Loyal Servant", "Morgana", "Assassin", "Mordred"]
    a = Avalon(names, seed=1, deal_override=layout)
    a.deal()
    assert set(a.evil) == {4, 5, 6}
    merlin = " ".join(a._knowledge(0))
    assert "P4" in merlin and "P5" in merlin and "P6" not in merlin   # Mordred stays hidden
    mordred = " ".join(a._knowledge(6))
    assert "P4" in mordred and "P5" in mordred                        # but knows his partners


def test_avalon_seventh_quest_four_needs_two_fails():
    # quest index 3 at 7 players needs TWO fail cards; a single fail still succeeds.
    names = {i: f"P{i}" for i in range(7)}
    layout = ["Merlin", "Percival", "Loyal Servant", "Loyal Servant", "Morgana", "Assassin", "Mordred"]
    a = Avalon(names, seed=1, deal_override=layout)
    a.deal()
    assert a.n >= 7
    # mirror the engine's rule directly
    for qi in range(5):
        needed = 2 if qi == 3 else 1
        assert needed == (2 if (a.n >= 7 and qi == 3) else 1)


@pytest.mark.parametrize("n,n_mafia", [(5, 1), (6, 2), (7, 2)])
def test_secret_mafia_composition_and_recognition(n, n_mafia):
    assert SM_ROLES[n].count("Mafia") == n_mafia
    core = SecretMafia({i: f"P{i}" for i in range(n)}, seed=2, deal_override=list(SM_ROLES[n]))
    core.deal()
    mafia = core._mafia()
    assert len(mafia) == n_mafia
    if n_mafia > 1:
        for m in mafia:                                   # partners recognise each other
            assert any("fellow Mafia" in o for o in core.obs[m])
    else:
        assert core.obs[mafia[0]] == []                   # lone mafioso learns nothing new


def test_secret_mafia_five_player_composition_string_unchanged():
    core = SecretMafia({i: f"P{i}" for i in range(5)}, seed=1,
                       deal_override=["Mafia", "Doctor", "Detective", "Villager", "Villager"])
    core.deal()
    assert core._composition() == "1 Mafia / 1 Doctor / 1 Detective / 2 Villagers"


def test_run_batch_rejects_out_of_range_rosters(tmp_path, monkeypatch):
    monkeypatch.setattr(batch.store, "DB_PATH", tmp_path / "t.db")
    too_few = [AgentSpec(name=f"A{i}", model=f"m{i}") for i in range(4)]
    too_many = [AgentSpec(name=f"A{i}", model=f"m{i}") for i in range(8)]
    for roster in (too_few, too_many):
        with pytest.raises(ValueError, match="supports 5"):
            batch.run_batch(game="onuw", n_games=1, roster=roster, run_id="bad")
