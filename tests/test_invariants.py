"""Invariant tests: vote tallying, win resolution, atomic vote, no-leak filtering."""
from __future__ import annotations

from arena.games.base import NO_KILL, compute_winners, tally_votes
from arena.games.onuw import ONUW
from tests.scripted import PolicyAgent, ScriptedDefault


# ---- all-tied-die + no-kill -------------------------------------------------
def test_tally_tie_all_die():
    # seats 0 and 2 each get 2 votes -> both die
    votes = {0: 2, 1: 2, 2: 0, 3: 0, 4: 1}
    assert tally_votes(votes, list(range(5))) == [0, 2]


def test_tally_no_consensus_spread():
    # every voter points at a distinct target, all at 1 -> nobody dies
    votes = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}
    assert tally_votes(votes, list(range(5))) == []


def test_tally_clear_majority():
    votes = {0: 3, 1: 3, 2: 3, 3: 0, 4: 0}
    assert tally_votes(votes, list(range(5))) == [3]


def test_tally_no_kill_plurality():
    votes = {0: NO_KILL, 1: NO_KILL, 2: NO_KILL, 3: 0, 4: 1}
    assert tally_votes(votes, list(range(5))) == []


# ---- win resolution ---------------------------------------------------------
def test_winners_village_kills_wolf():
    roles = {0: "Werewolf", 1: "Villager", 2: "Seer", 3: "Villager", 4: "Robber"}
    w = compute_winners(roles, deaths=[0], votes={})
    assert w["village"] and not w["werewolf"]


def test_winners_wolves_survive():
    roles = {0: "Werewolf", 1: "Villager", 2: "Seer", 3: "Villager", 4: "Robber"}
    w = compute_winners(roles, deaths=[1], votes={})
    assert w["werewolf"] and not w["village"]


def test_winners_no_wolf_in_play():
    # both wolves in center -> village wins ONLY if nobody dies
    roles = {0: "Villager", 1: "Seer", 2: "Robber", 3: "Villager", 4: "Troublemaker"}
    assert compute_winners(roles, deaths=[], votes={})["village"]
    assert not compute_winners(roles, deaths=[2], votes={})["village"]


def test_winners_hunter_chain():
    roles = {0: "Hunter", 1: "Werewolf", 2: "Villager", 3: "Villager", 4: "Seer"}
    w = compute_winners(roles, deaths=[0], votes={0: 1})  # Hunter dies, dragged its vote (the wolf)
    assert 1 in w["deaths"] and w["village"]


def test_winners_tanner():
    roles = {0: "Tanner", 1: "Werewolf", 2: "Villager", 3: "Villager", 4: "Seer"}
    w = compute_winners(roles, deaths=[0], votes={})
    assert w["tanner"] and not w["werewolf"] and not w["village"]


# ---- atomic vote: all voters see the same frozen pre-vote state -------------
def test_vote_atomicity_frozen_state():
    record = []
    names = {i: f"P{i}" for i in range(5)}
    agents = {i: PolicyAgent(record=record) for i in range(5)}
    ONUW(names, seed=3).play(agents)
    assert len(record) == 5  # one vote-phase observation per player
    convs = []
    for obs in record:
        marker = "Conversation and events so far:"
        convs.append(obs.split(marker, 1)[1] if marker in obs else "")
    # every voter saw an identical public transcript — nobody saw another's vote
    assert all(c == convs[0] for c in convs)
    assert all("votes" not in c.replace("voted", "") for c in convs)


# ---- no-leak filtering: a villager learns nothing private -------------------
def test_no_leak_villager_has_no_night_info():
    # seat 0 forced Villager; verify it learns nothing and its prompt hides others' roles
    layout = ["Villager", "Werewolf", "Seer", "Robber", "Troublemaker", "Werewolf", "Villager", "Minion"]
    names = {i: f"P{i}" for i in range(5)}
    core = ONUW(names, seed=1, deal_override=layout)
    core.deal()
    core.run_night({i: ScriptedDefault() for i in range(5)})
    assert core.obs[0] == []  # a plain villager has zero private observations
    # the seer (seat 2) only learned what it looked at (center by default), not players
    assert core.obs[2] and all("center" in o for o in core.obs[2])
