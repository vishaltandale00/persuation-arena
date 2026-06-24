"""ONUW conformance: deal, role actions, swaps, recognition, Doppelganger, full game."""
from __future__ import annotations

from arena.games.base import team_of
from arena.games.onuw import DEFAULT_DECK, ONUW
from tests.scripted import ScriptedDefault

NAMES = {i: f"P{i}" for i in range(5)}


def _night(layout, seed=1):
    core = ONUW(NAMES, seed=seed, deal_override=layout)
    core.deal()
    core.run_night({i: ScriptedDefault() for i in range(5)})
    return core


def test_default_deck_composition():
    assert DEFAULT_DECK.count("Werewolf") == 2
    assert len(DEFAULT_DECK) == 8


def test_deal_reproducible_by_seed():
    a = ONUW(NAMES, seed=42); a.deal()
    b = ONUW(NAMES, seed=42); b.deal()
    assert a.dealt == b.dealt and a.center == b.center


def test_robber_swaps_and_only_robber_knows():
    # Robber at seat0, Werewolf at seat1; default robs seat1
    layout = ["Robber", "Werewolf", "Villager", "Seer", "Villager", "Troublemaker", "Villager", "Minion"]
    c = _night(layout)
    assert c.current[0] == "Werewolf"          # robber took the wolf card
    assert c.current[1] == "Robber"            # victim now holds Robber
    assert c.believes[0] == "Werewolf"         # robber KNOWS its new role
    assert c.believes[1] == "Werewolf"         # victim still believes its dealt role


def test_troublemaker_swaps_without_learning():
    # Troublemaker at seat0; default swaps seats 1 and 2
    layout = ["Troublemaker", "Werewolf", "Seer", "Villager", "Villager", "Robber", "Villager", "Minion"]
    c = _night(layout)
    assert c.current[1] == "Seer" and c.current[2] == "Werewolf"   # 1<->2 swapped
    assert c.believes[1] == "Werewolf" and c.believes[2] == "Seer"  # neither learns


def test_seer_views_center_by_default():
    layout = ["Seer", "Werewolf", "Villager", "Villager", "Robber", "Troublemaker", "Minion", "Villager"]
    c = _night(layout)
    # center = layout[5:8] = Troublemaker, Minion, Villager; default views #1,#2
    joined = " ".join(c.obs[0])
    assert "Troublemaker" in joined and "Minion" in joined


def test_minion_knows_wolves_but_wolves_do_not_know_minion():
    layout = ["Minion", "Werewolf", "Villager", "Seer", "Robber", "Villager", "Villager", "Troublemaker"]
    c = _night(layout)
    assert any("P1" in o for o in c.obs[0])            # minion saw the wolf (seat1=P1)
    assert all("Minion" not in o for o in c.obs[1])    # wolf never learned the minion


def test_doppelganger_copies_and_acts():
    # Doppelganger at seat0; default copies seat1 (Seer) and performs seer action.
    # No Robber/Troublemaker in play, so nothing swaps the doppelganger afterwards.
    layout = ["Doppelganger", "Seer", "Villager", "Villager", "Villager", "Werewolf", "Werewolf", "Minion"]
    c = _night(layout)
    assert c._doppel_role[0] == "Seer"
    assert c.current[0] == "Seer"
    assert any("copied" in o for o in c.obs[0])
    assert any("center" in o for o in c.obs[0])  # performed the copied Seer's center look


def test_full_game_completes_and_is_consistent():
    core = ONUW(NAMES, seed=11)
    rec = core.play({i: ScriptedDefault() for i in range(5)})
    assert [p["name"] for p in rec["phases"]] == ["Night", "Discussion", "Vote", "Result"]
    assert rec["winner_team"] in ("good", "evil", "void")  # "void" = no evil faction was in play
    # every player's team is well-formed and won-flag is a bool
    for p in rec["players"]:
        assert p["team"] in ("good", "evil")
        assert isinstance(p["won"], bool)
    # at least the cards in play are the full deck of 8
    assert len(rec["cardsInPlay"]) == 8 and len(rec["center"]) == 3
