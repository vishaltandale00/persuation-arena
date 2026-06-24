"""Shared game-core helpers: the Agent protocol, team map, and vote tallying."""
from __future__ import annotations

from typing import Any, Callable, Protocol


class Agent(Protocol):
    name: str
    model: str

    def act(self, observation: str, parse_action: Callable[[Any, str], Any],
            default_action: Any, **turn_meta: Any): ...


# Role -> team. Evil roles across games; everything else is good/village.
EVIL_ROLES = {"Werewolf", "Minion",                      # ONUW
              "Morgana", "Assassin", "Mordred", "Oberon",  # Avalon
              "Mafia"}                                      # Secret Mafia


def team_of(role: str) -> str:
    return "evil" if role in EVIL_ROLES else "good"


def agent_stats(agent: Any) -> dict:
    """Per-seat reliability telemetry for an agent over one game.

    Reads the agent's `calls` log (a list of {ok, ms, raw} records appended per turn).
    Scripted/test agents without a `calls` attribute report zeros. `forfeits` counts turns
    where the model failed and the engine substituted a default action — non-skill noise that
    scoring can surface or exclude.
    """
    calls = getattr(agent, "calls", []) or []
    forfeits = sum(1 for c in calls if not c.get("ok", True))
    return {"calls": len(calls), "forfeits": forfeits}


NO_KILL = -1  # vote target meaning "point up / no one"


def tally_votes(votes: dict[int, int], alive: list[int]) -> list[int]:
    """Return the list of players who die.

    Plurality with all-tied-die. NO_KILL votes count toward 'no one'. If every voter spread to a
    distinct target and the max is 1 (no consensus), nobody dies — the ONUW 'everyone got one vote'
    case. If 'no one' is (tied for) the plurality, nobody dies.
    """
    if not votes:
        return []
    counts: dict[int, int] = {}
    for tgt in votes.values():
        counts[tgt] = counts.get(tgt, 0) + 1
    top = max(counts.values())
    if top == 1 and len(counts) == len(votes):
        return []  # complete spread, no consensus
    winners = [pid for pid, c in counts.items() if c == top]
    if NO_KILL in winners:
        return []  # the table chose (or tied on) no elimination
    return sorted(winners)


def compute_winners(roles: dict[int, str], deaths: list[int], votes: dict[int, int]) -> dict:
    """Pure ONUW win resolution on END-OF-NIGHT roles.

    roles: pid -> end-of-night role. deaths: players killed by the vote. votes: pid -> target
    (for the Hunter chain). Returns independent team-win booleans + the final death set.
    """
    D = set(deaths)
    # Hunter chain: a dead Hunter drags down whoever it voted for, into the same death set.
    changed = True
    while changed:
        changed = False
        for h in list(D):
            if roles.get(h) == "Hunter":
                t = votes.get(h)
                if t is not None and t >= 0 and t in roles and t not in D:
                    D.add(t)
                    changed = True

    wolves = [p for p, r in roles.items() if r == "Werewolf"]
    minion_exists = any(r == "Minion" for r in roles.values())
    wolf_died = any(roles[p] == "Werewolf" for p in D)
    tanner_died = any(roles[p] == "Tanner" for p in D)

    if wolves:
        village_win = wolf_died
        werewolf_win = (not wolf_died) and (not tanner_died)
    else:
        village_win = len(D) == 0
        werewolf_win = (len(D) > 0) and minion_exists and (not tanner_died)
    tanner_win = tanner_died
    # No team met a win condition. The only way to reach this is: no Werewolf and no Minion at
    # end-of-night, a death occurred, and no Tanner died — i.e. no evil faction was in play and
    # the vote killed an innocent. That is a no-contest, not an evil win.
    no_contest = not (village_win or werewolf_win or tanner_win)
    return {"village": village_win, "werewolf": werewolf_win, "tanner": tanner_win,
            "no_contest": no_contest, "deaths": sorted(D)}


def is_no_contest(seats: list[dict]) -> bool:
    """True for a stored game with no evil seat AND no winning seat — the ONUW degenerate where no
    Werewolf or Minion was in play and the vote eliminated an innocent. There was no opposing
    faction, so it is a no-contest: excluded from win-rate scoring and the rating replay (rating a
    seat for a loss against nobody would corrupt the Elo). A no-wolf game where the village correctly
    idles to a win still has winning seats, so it is kept. Avalon/Mafia always deal an evil faction,
    so this only ever trips on ONUW."""
    no_evil = not any(s.get("team") == "evil" for s in seats)
    no_winner = not any(int(s.get("won") or 0) for s in seats)
    return no_evil and no_winner


def player_won(role: str, wins: dict) -> bool:
    if role in ("Werewolf", "Minion"):
        return wins["werewolf"]
    if role == "Tanner":
        return wins["tanner"]
    return wins["village"]
