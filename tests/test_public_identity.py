from __future__ import annotations

from arena.games.avalon import Avalon
from arena.games.onuw import ONUW
from arena.games.secret_mafia import SecretMafia
from arena.identity import public_ref, validate_unique_public_names


class _Resp:
    def __init__(self, action):
        self.action = action
        self.reasoning = "ok"
        self.ms = 0.0


class CaptureAgent:
    name = "capture"
    model = "scripted"

    def __init__(self, raw_action):
        self.raw_action = raw_action
        self.observation = ""
        self.meta = {}

    def act(self, observation, parse_action, default_action, **turn_meta):
        self.observation = observation
        self.meta = turn_meta
        return _Resp(parse_action(self.raw_action, ""))


def _names(n=5):
    base = ["Alice", "Bob", "Cora", "Dane", "Eve", "Finn", "Gail"]
    return {i: base[i] for i in range(n)}


def test_onuw_prompt_and_legal_actions_use_public_refs():
    names = _names()
    layout = ["Robber", "Werewolf", "Seer", "Tanner", "Villager", "Mason", "Mason", "Drunk"]
    game = ONUW(names, seed=1, deal_override=layout, deck=list(layout))
    game.deal()

    prompt = game.base_prompt(0)
    assert "@alice" in prompt and "@bob" in prompt
    assert "seat 0" not in prompt and "seat 1" not in prompt

    agent = CaptureAgent({"target": "@bob"})
    game._robber_action(0, agent)

    assert "seat 0" not in agent.observation and "seat 1" not in agent.observation
    legal = agent.meta["legal_action"]
    assert legal["schema"]["properties"]["target"]["type"] == ["string", "null"]
    assert "@bob" in legal["schema"]["properties"]["target"]["enum"]
    assert {"name": "Bob", "ref": "@bob"} in legal["choices"]["players"]
    assert all("seat" not in p for p in legal["choices"]["players"])


def test_other_game_prompts_use_public_refs_not_seats():
    names = _names()

    avalon = Avalon(names, seed=1, deal_override=["Merlin", "Percival", "Loyal Servant", "Morgana", "Assassin"])
    avalon.deal()
    avalon_prompt = avalon.base_prompt(0)
    assert "@alice" in avalon_prompt and "@bob" in avalon_prompt
    assert "seat 0" not in avalon_prompt and "seat 1" not in avalon_prompt

    mafia = SecretMafia(names, seed=1, deal_override=["Mafia", "Doctor", "Detective", "Villager", "Villager"])
    mafia.deal()
    mafia_prompt = mafia.base_prompt(0)
    assert "@alice" in mafia_prompt and "@bob" in mafia_prompt
    assert "seat 0" not in mafia_prompt and "seat 1" not in mafia_prompt


def test_public_refs_use_ascii_separator_normalization():
    assert public_ref("José") == "@jos"
    assert public_ref("Jose") == "@jose"
    assert validate_unique_public_names(["José", "Jose"]) is None
