"""ONUW conformance: deal, role actions, swaps, recognition, Doppelganger, full game."""
from __future__ import annotations

import json

from arena.games.base import team_of
from arena.games.onuw import DEFAULT_DECK, ONUW
from tests.scripted import ScriptedDefault

NAMES = {i: f"P{i}" for i in range(5)}


class RecordingAgent:
    name = "R"
    model = "scripted"

    def __init__(self):
        self.observations: list[str] = []
        self.turn_meta: list[dict] = []

    def act(self, observation, parse_action, default_action, **turn_meta):
        self.observations.append(observation)
        self.turn_meta.append(turn_meta)

        class Resp:
            reasoning = "(recorded)"
            action = default_action
            ms = 0.0

        return Resp()


def _rules_payload_from_prompt(prompt: str) -> dict:
    start = "ONUW_RULES_PAYLOAD_BEGIN"
    end = "ONUW_RULES_PAYLOAD_END"
    assert start in prompt and end in prompt
    return json.loads(prompt.split(start, 1)[1].split(end, 1)[0].strip())


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


def test_runtime_prompt_includes_comprehensive_onuw_rules_payload():
    layout = ["Robber", "Doppelganger", "Werewolf", "Troublemaker", "Drunk", "Minion", "Hunter", "Tanner"]
    core = ONUW(NAMES, seed=7, deck=list(layout), deal_override=list(layout))
    core.deal()
    agent = RecordingAgent()

    core._robber_action(0, agent)

    prompt = agent.observations[-1]
    payload = _rules_payload_from_prompt(prompt)
    assert payload["game_setup"]["player_count"] == 5
    assert payload["game_setup"]["total_cards"] == 8
    assert payload["game_setup"]["dealt_player_cards"] == 5
    assert payload["game_setup"]["center_cards"] == 3
    assert payload["game_setup"]["deck_counts"]["Werewolf"] == 1
    assert payload["role_state_distinctions"]["dealt_role"]
    assert payload["role_state_distinctions"]["final_role"]
    assert payload["night_action_order"] == [
        "Doppelganger", "Werewolf", "Minion", "Mason", "Seer", "Robber",
        "Troublemaker", "Drunk", "Insomniac",
    ]
    assert "immediately performs the copied action" in payload["doppelganger_caveat"]
    assert "sees and believes the new card" in payload["action_rules"]["robber"]
    assert "without looking" in payload["action_rules"]["troublemaker"]
    assert "does not look at the new card" in payload["action_rules"]["drunk"]
    assert payload["discussion_rules"]["visibility"].startswith("Discussion is public")
    assert "simultaneous" in payload["vote_rules"]["timing"]
    assert "target -1" in payload["vote_rules"]["legal_targets"]
    for key in ("tanner", "hunter", "minion", "werewolf", "village"):
        assert payload["win_conditions"][key]
    assert payload["current_step"] == {"phase": "night", "action_kind": "onuw.robber.swap_or_decline"}


def test_rules_payload_does_not_leak_other_dealt_roles_or_center_identities():
    layout = ["Seer", "Werewolf", "Minion", "Robber", "Tanner", "Hunter", "Drunk", "Villager"]
    core = ONUW(NAMES, seed=9, deck=list(layout), deal_override=list(layout))
    core.deal()

    prompt = core.base_prompt(0, phase="discussion", action_kind="onuw.discussion.speak_or_pass")
    payload = _rules_payload_from_prompt(prompt)

    assert "P1(seat 1)" in prompt
    assert payload["game_setup"]["deck_counts"] == {
        "Drunk": 1,
        "Hunter": 1,
        "Minion": 1,
        "Robber": 1,
        "Seer": 1,
        "Tanner": 1,
        "Villager": 1,
        "Werewolf": 1,
    }
    assert "P1: Werewolf" not in prompt
    assert "seat 1: Werewolf" not in prompt
    assert "P2: Minion" not in prompt
    assert "center #1: Hunter" not in prompt
    assert "center #2: Drunk" not in prompt
    assert "center #3: Villager" not in prompt


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
