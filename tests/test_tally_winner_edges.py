"""Edge-gap tests for base.tally_votes / compute_winners / team_of.

Complements tests/test_invariants.py (which covers the all-tied-die, no-consensus
spread, clear majority, sole no-kill plurality, single-hop Hunter chain, and Tanner
cases). Here we pin the remaining edges: empty/degenerate tallies, a no-kill TIED for
plurality, the full evil-role team map, a TRANSITIVE multi-hop Hunter chain, the
Tanner+wolf independent-boolean case, and a Minion-only (no wolf) evil win.
"""
from __future__ import annotations

from arena.games.base import NO_KILL, compute_winners, tally_votes, team_of


# ---- tally_votes edges ------------------------------------------------------
def test_tally_empty_votes_kills_no_one():
    assert tally_votes({}, [0, 1, 2]) == []


def test_tally_single_vote_no_kill():
    # one voter, one target -> top==1 and len(counts)==len(votes): complete spread, nobody dies
    assert tally_votes({0: 1}, [0, 1]) == []


def test_tally_two_voter_spread_no_kill():
    # the documented "everyone got one vote" case at n=2: distinct targets, max 1 -> nobody dies
    assert tally_votes({0: 1, 1: 0}, [0, 1]) == []


def test_tally_no_kill_tied_for_plurality():
    # NO_KILL and seat 0 are BOTH at 2 (tied plurality). NO_KILL in winners -> nobody dies.
    votes = {0: NO_KILL, 1: NO_KILL, 2: 0, 3: 0, 4: 1}
    assert tally_votes(votes, list(range(5))) == []


# ---- team_of role map -------------------------------------------------------
def test_team_of_evil_roles():
    for role in ("Mafia", "Mordred", "Oberon", "Minion", "Werewolf"):
        assert team_of(role) == "evil"


def test_team_of_unknown_role_is_good():
    assert team_of("Villager") == "good"
    assert team_of("Plumber") == "good"  # any role not in EVIL_ROLES falls to good


# ---- compute_winners edges --------------------------------------------------
def test_winners_transitive_hunter_chain():
    # 0 (Hunter) voted 1 (also Hunter) who voted 2 (the wolf). One eliminated Hunter
    # should drag the WHOLE chain into the death set: {0,1,2}, and killing the wolf is a village win.
    roles = {0: "Hunter", 1: "Hunter", 2: "Werewolf", 3: "Villager", 4: "Seer"}
    w = compute_winners(roles, deaths=[0], votes={0: 1, 1: 2})
    assert w["deaths"] == [0, 1, 2]
    assert w["village"] and not w["werewolf"]


def test_winners_tanner_and_wolf_both_die_independent_booleans():
    # Wolf present and dies -> village wins; Tanner also dies -> tanner wins. The two booleans
    # are independent: village True AND tanner True, werewolf False (a dead wolf can't win).
    roles = {0: "Tanner", 1: "Werewolf", 2: "Villager", 3: "Seer", 4: "Robber"}
    w = compute_winners(roles, deaths=[0, 1], votes={})
    assert w["village"] is True
    assert w["tanner"] is True
    assert w["werewolf"] is False
    assert w["no_contest"] is False


def test_winners_minion_only_evil_is_a_win_not_no_contest():
    # No Werewolf in play but a Minion is. A non-Tanner death -> the lone-evil Minion WINS;
    # this is a genuine contest, NOT the no-evil-faction void.
    roles = {0: "Minion", 1: "Villager", 2: "Seer", 3: "Villager", 4: "Robber"}
    w = compute_winners(roles, deaths=[1], votes={})
    assert w["werewolf"] is True       # minion/wolf share the evil-win flag
    assert w["village"] is False
    assert w["no_contest"] is False
