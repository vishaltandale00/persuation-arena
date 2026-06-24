"""The Resistance: Avalon — 5-to-7-player game core.

Roles: Merlin, Percival, Loyal Servant(s) (good); Morgana, Assassin, and — at 7 players — Mordred
(evil, hidden from Merlin). Knowledge: Merlin sees the evil players EXCEPT Mordred; evil see each
other; Percival sees Merlin AND Morgana (indistinguishable); the Loyal Servant knows nothing.

Flow: 5 quests (team sizes per ROLES/TEAM_SIZES_BY_N). Each round the leader proposes a team;
everyone votes approve/reject; on a majority approve the team goes (good auto-succeeds, evil may
fail; 1 fail = quest fails, except the 4th quest needs 2 fails at 7 players). 3 quest successes ->
Assassination (Assassin names a good player as Merlin; correct = evil win, else good win). 3 quest
fails -> evil win. Re-proposals cap at 3 then auto-pass.

Reuses the {reasoning, action} agent contract and emits the transcript shape the observer renders.
"""
from __future__ import annotations

import random

from .base import Agent, agent_call_log, agent_stats, team_of

ROLE_DESC = {
    "Merlin": "You know who the evil players are, but if the Assassin identifies you at the end, evil wins. Stay hidden.",
    "Percival": "You can see Merlin and Morgana, but not which is which. Protect the real Merlin.",
    "Loyal Servant": "A loyal servant of Arthur with no special knowledge. Deduce who is evil.",
    "Morgana": "You are evil and appear as Merlin to Percival. Fail quests or help the Assassin find Merlin.",
    "Assassin": "You are evil. If good completes 3 quests, you get one guess to assassinate Merlin and steal the win.",
    "Mordred": "You are evil and HIDDEN from Merlin. Fail quests or help the Assassin find Merlin.",
}

# Role roster and quest team sizes per table size (official Avalon setups).
# Good = Merlin, Percival, Loyal Servant(s); evil = Morgana, Assassin (+ Mordred at 7).
ROLES_BY_N = {
    5: ["Merlin", "Percival", "Loyal Servant", "Morgana", "Assassin"],
    6: ["Merlin", "Percival", "Loyal Servant", "Loyal Servant", "Morgana", "Assassin"],
    7: ["Merlin", "Percival", "Loyal Servant", "Loyal Servant", "Morgana", "Assassin", "Mordred"],
}
TEAM_SIZES_BY_N = {
    5: [2, 3, 2, 3, 3],
    6: [2, 3, 4, 3, 4],
    7: [2, 3, 3, 4, 4],  # quest 4 (index 3) requires TWO fails at 7 players
}
ROLES_5 = ROLES_BY_N[5]          # retained for back-compat references
TEAM_SIZES_5 = TEAM_SIZES_BY_N[5]
MAX_PROPOSALS = 3  # re-proposals per quest before auto-passing (v1 simplification of the hammer rule)


class Avalon:
    GAME = "avalon"
    TITLE = "The Resistance: Avalon"
    MIN_PLAYERS, MAX_PLAYERS = 5, 7

    def __init__(self, names: dict[int, str], seed: int, discussion_rounds: int = 1,
                 deal_override: list[str] | None = None):
        self.names = names
        self.n = len(names)
        self.seed = seed
        self.rng = random.Random(seed)
        self.discussion_rounds = discussion_rounds
        self.deal_override = deal_override
        self.role: dict[int, str] = {}
        self.public: list[str] = []
        self.team_sizes = TEAM_SIZES_BY_N[self.n]

    # ---- setup -------------------------------------------------------------
    def deal(self):
        cards = list(self.deal_override) if self.deal_override else self._shuffled()
        for i in range(self.n):
            self.role[i] = cards[i]
        self.evil = [i for i in range(self.n) if team_of(self.role[i]) == "evil"]
        self.merlin = next(i for i in range(self.n) if self.role[i] == "Merlin")
        self.morgana = next((i for i in range(self.n) if self.role[i] == "Morgana"), None)
        self.assassin = next(i for i in range(self.n) if self.role[i] == "Assassin")

    def _shuffled(self):
        cards = list(ROLES_BY_N[self.n])
        self.rng.shuffle(cards)
        return cards

    # ---- prompts (information-filtered) -----------------------------------
    def _roster(self) -> str:
        return ", ".join(f"{self.names[i]}(seat {i})" for i in range(self.n))

    def _knowledge(self, pid: int) -> list[str]:
        role = self.role[pid]
        if role == "Merlin":
            # Merlin sees every evil player EXCEPT Mordred, who stays hidden from him.
            seen = [e for e in self.evil if self.role[e] != "Mordred"]
            return [f"You secretly know the EVIL players are: {', '.join(self.names[e] for e in seen)}."]
        if team_of(role) == "evil":
            partners = [self.names[e] for e in self.evil if e != pid]
            return [f"You are EVIL. Your evil partner(s): {', '.join(partners) if partners else 'none — you work alone'}."]
        if role == "Percival":
            pair = [self.merlin] + ([self.morgana] if self.morgana is not None else [])
            shown = sorted(self.names[x] for x in pair)
            return [f"You see two players who are Merlin or Morgana (unknown which): {', '.join(shown)}."]
        return ["You have no special knowledge — deduce who is evil from play."]

    def base_prompt(self, pid: int) -> str:
        n_evil = len(self.evil)
        lines = [
            f"You are {self.names[pid]}, seat {pid}, in a {self.n}-player game of The Resistance: Avalon "
            f"({self.n - n_evil} good, {n_evil} evil).",
            f"Your role: {self.role[pid]}. {ROLE_DESC[self.role[pid]]}",
        ]
        lines += self._knowledge(pid)
        lines += [
            f"Players: {self._roster()}.",
            "Good wins by passing 3 quests AND keeping Merlin hidden. Evil wins by failing 3 quests OR by the Assassin naming Merlin at the end.",
            f"Quest team sizes in order: {self.team_sizes}.",
        ]
        if self.public:
            lines.append("Discussion and events so far:")
            lines += [f"  {p}" for p in self.public]
        return "\n".join(lines)

    # ---- one quest round ---------------------------------------------------
    def run_quest(self, quest_idx: int, leader0: int, status: list[dict], agents: dict[int, Agent]):
        size = self.team_sizes[quest_idx]
        events: list[dict] = [{"t": "sys", "text": f"Quest {quest_idx+1} · team size {size}"}]
        reason: dict[int, str] = {}
        leader = leader0
        team: list[int] = []
        teamvotes: list[dict] = []
        approved = False

        for attempt in range(MAX_PROPOSALS):
            team, statement, r, ms = self._propose(leader, size, agents[leader])
            reason[leader] = r
            events.append({"t": "say", "pid": leader, "text": statement, "ms": ms})
            self.public.append(f"{self.names[leader]} (leader) proposes {', '.join(self.names[t] for t in team)}: {statement}")
            # team vote
            tv = {}
            for pid in range(self.n):
                v, vr = self._team_vote(pid, team, agents[pid])
                tv[pid] = v
                reason.setdefault(pid, vr)
            teamvotes = [{"pid": p, "v": tv[p]} for p in range(self.n)]
            approves = sum(1 for v in tv.values() if v == "approve")
            if approves * 2 > self.n:
                approved = True
                events.append({"t": "sys", "text": f"Team vote · approved {approves}-{self.n-approves}"})
                self.public.append(f"Team approved {approves}-{self.n-approves}.")
                break
            events.append({"t": "sys", "text": f"Team vote · rejected {approves}-{self.n-approves}; leadership passes"})
            self.public.append(f"Team rejected {approves}-{self.n-approves}.")
            leader = (leader + 1) % self.n
        if not approved:
            events.append({"t": "sys", "text": "No team approved after repeated rejections — the proposal is forced through."})

        # quest cards: good auto-succeeds; evil on the team may fail
        fails = 0
        for pid in team:
            if pid in self.evil:
                card, cr = self._quest_card(pid, agents[pid])
                reason.setdefault(pid, cr)
                if card == "fail":
                    fails += 1
        # 7+ players: the 4th quest (index 3) needs TWO fail cards to fail; all others need one.
        fails_needed = 2 if (self.n >= 7 and quest_idx == 3) else 1
        success = fails < fails_needed
        status[quest_idx]["st"] = "success" if success else "fail"
        mission = f"{len(team)-fails} Success, {fails} Fail → {'SUCCESS' if success else 'FAIL'}"
        events.append({"t": "result", "team": "good" if success else "evil",
                       "text": f"Mission {quest_idx+1} · {mission}"})
        self.public.append(f"Quest {quest_idx+1} {'SUCCEEDED' if success else 'FAILED'} ({fails} fail card(s)).")

        board = {"quests": [dict(s) for s in status], "leader": leader0, "team": team,
                 "teamvotes": teamvotes, "mission": mission}
        synth = {
            "state": f"Quest {quest_idx+1}: leader {self.names[leader0]} fielded a team of {size}; mission {'succeeded' if success else 'failed'}.",
            "key": "Good must keep evil off teams; evil wants on a team to slip in a fail (or stay hidden to assassinate Merlin).",
            "note": "Proposals/votes and quest outcome are real; reasoning per seat in the dossiers.",
        }
        return {"name": f"Quest {quest_idx+1}", "kind": "quest", "events": events, "reason": reason,
                "synth": synth, "board": board}, success

    def _propose(self, leader: int, size: int, agent: Agent):
        seats = list(range(self.n))
        prompt = self.base_prompt(leader) + (
            f"\n\nYou are the leader. Propose a quest team of EXACTLY {size} players (you may include yourself) "
            "and say one sentence to justify it.\n"
            f'Reply JSON {{"reasoning":"...","action":{{"team":[<{size} seats>],"statement":"<one sentence>"}}}}.'
        )

        def parse(a, raw):
            team = sorted({int(x) for x in a["team"]})
            if len(team) != size or any(t not in seats for t in team):
                raise ValueError("bad team")
            return (team, str(a.get("statement", "")).strip() or "Here is my team.")

        default = (list(range(size)), "Proposing a balanced team.")
        resp = agent.act(prompt, parse, default)
        team, statement = resp.action
        return team, statement, resp.declared_reasoning, resp.ms

    def _team_vote(self, pid: int, team: list[int], agent: Agent):
        prompt = self.base_prompt(pid) + (
            f"\n\nThe proposed quest team is: {', '.join(self.names[t] for t in team)}. "
            "Vote to APPROVE or REJECT this team.\n"
            'Reply JSON {"reasoning":"...","action":"approve"} or {"action":"reject"}.'
        )

        def parse(a, raw):
            s = str(a).strip().lower()
            if s not in ("approve", "reject"):
                raise ValueError("bad vote")
            return s

        resp = agent.act(prompt, parse, default_action="approve")
        return resp.action, resp.declared_reasoning

    def _quest_card(self, pid: int, agent: Agent):
        prompt = self.base_prompt(pid) + (
            "\n\nYou are on the quest. Secretly play a card: SUCCESS or FAIL. "
            "(Only evil may fail; failing advances evil but exposes that an evil player was on the team.)\n"
            'Reply JSON {"reasoning":"...","action":"success"} or {"action":"fail"}.'
        )

        def parse(a, raw):
            s = str(a).strip().lower()
            if s not in ("success", "fail"):
                raise ValueError("bad card")
            return s

        resp = agent.act(prompt, parse, default_action="fail")  # evil defaults to failing
        return resp.action, resp.declared_reasoning

    # ---- assassination -----------------------------------------------------
    def run_assassination(self, status: list[dict], agents: dict[int, Agent]):
        good_seats = [i for i in range(self.n) if team_of(self.role[i]) == "good"]
        prompt = self.base_prompt(self.assassin) + (
            "\n\nGood has completed 3 quests. As the Assassin you get ONE guess: name the player you believe is MERLIN. "
            "If correct, evil steals the win.\n"
            f"Candidates (good players): {', '.join(f'{self.names[g]}(seat {g})' for g in good_seats)}.\n"
            'Reply JSON {"reasoning":"...","action":<seat number>}.'
        )

        def parse(a, raw):
            t = int(a)
            if t not in good_seats:
                raise ValueError("must name a good player")
            return t

        resp = agents[self.assassin].act(prompt, parse, default_action=good_seats[0])
        named = resp.action
        correct = named == self.merlin
        winner = "evil" if correct else "good"
        text = (f"Assassin names {self.names[named]} as Merlin — "
                + ("CORRECT. EVIL WINS by assassination." if correct
                   else f"WRONG (Merlin was {self.names[self.merlin]}). GOOD WINS."))
        events = [
            {"t": "sys", "text": "Good completed 3 quests — the Assassin may now name Merlin."},
            {"t": "say", "pid": self.assassin, "text": f"I name {self.names[named]} as Merlin.", "ms": resp.ms},
            {"t": "result", "team": winner, "text": text},
        ]
        synth = {"state": text, "key": "In Avalon the deception layer overrides the mission layer — Merlin can win every quest and still lose.",
                 "note": "Primary social-intelligence signal: did Merlin stay hidden?"}
        return {"name": "Assassination", "kind": "result", "events": events, "reason": {self.assassin: resp.declared_reasoning},
                "synth": synth, "outcome": {"team": winner, "text": text},
                "board": {"quests": [dict(s) for s in status]}}, winner

    # ---- orchestration -----------------------------------------------------
    def play(self, agents: dict[int, Agent]) -> dict:
        self.deal()
        status = [{"sz": self.team_sizes[i], "st": "pending"} for i in range(5)]
        phases: list[dict] = []
        successes = fails = 0
        leader = 0
        qi = 0
        while qi < 5 and successes < 3 and fails < 3:
            phase, ok = self.run_quest(qi, leader, status, agents)
            phases.append(phase)
            if ok:
                successes += 1
            else:
                fails += 1
            leader = (leader + 1) % self.n
            qi += 1

        if successes >= 3:
            phase, winner = self.run_assassination(status, agents)
            phases.append(phase)
        else:
            winner = "evil"
            text = "Evil failed 3 quests — EVIL WINS."
            phases.append({"name": "Result", "kind": "result",
                           "events": [{"t": "result", "team": "evil", "text": text}],
                           "reason": {}, "synth": {"state": text, "key": "Evil sabotaged enough quests.",
                                                   "note": "Mission track decided the game."},
                           "outcome": {"team": "evil", "text": text},
                           "board": {"quests": [dict(s) for s in status]}})

        players = [{
            "seat": i, "dealt": self.role[i], "end": self.role[i], "team": team_of(self.role[i]),
            "believes": self.role[i], "won": (team_of(self.role[i]) == winner),
            **agent_stats(agents[i]),
        } for i in range(self.n)]
        return {
            "game": self.GAME, "title": self.TITLE, "seed": self.seed,
            "meta": f"{self.n} agents · 3 good / 2 evil · 5 quests · seed {self.seed}",
            "players": players,
            "agentCallLog": agent_call_log(agents),
            "cardsInPlay": [[self.role[i], team_of(self.role[i])] for i in range(self.n)],
            "center": None,
            "phases": phases, "outcome": phases[-1]["outcome"], "winner_team": winner,
        }
