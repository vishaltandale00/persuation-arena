"""Secret Mafia — compact native core (the free third game).

5–7 players: 1 Doctor + 1 Detective, with Mafia scaling up (1 at five players, 2 at six/seven) and
the rest Villagers (see ROLES_BY_N). When more than one Mafia is dealt they recognise each other.
Repeats Night (mafia kills, doctor protects, detective investigates) and Day (round-robin
discussion + elimination vote) until the Mafia is eliminated (village wins) or the Mafia reaches
parity (mafia wins). Reuses the {reasoning, action} agent contract and emits the observer shape.
"""
from __future__ import annotations

import random
from collections import Counter

from arena.identity import participant_label, seat_for_participant_ref
from .base import Agent, agent_call_log, agent_stats, team_of

ROLE_DESC = {
    "Mafia": "You secretly kill one player each night. Win when the Mafia equals the number of remaining villagers.",
    "Doctor": "Each night you protect one player from the Mafia. Win by eliminating the Mafia.",
    "Detective": "Each night you investigate one player and learn if they are Mafia. Win by eliminating the Mafia.",
    "Villager": "You have no night action. Win by deducing and voting out the Mafia.",
}
ROLES_5 = ["Mafia", "Doctor", "Detective", "Villager", "Villager"]
# Role roster per table size: 1 Doctor + 1 Detective always, Mafia scale up, the rest are Villagers.
ROLES_BY_N = {
    5: ["Mafia", "Doctor", "Detective", "Villager", "Villager"],
    6: ["Mafia", "Mafia", "Doctor", "Detective", "Villager", "Villager"],
    7: ["Mafia", "Mafia", "Doctor", "Detective", "Villager", "Villager", "Villager"],
}
MAX_ROUNDS = 4


class SecretMafia:
    GAME = "secret_mafia"
    TITLE = "Secret Mafia"
    MIN_PLAYERS, MAX_PLAYERS = 5, 7

    def __init__(self, names, seed, discussion_rounds=1, deal_override=None):
        self.names = names
        self.n = len(names)
        self.seed = seed
        self.rng = random.Random(seed)
        self.discussion_rounds = discussion_rounds
        self.deal_override = deal_override
        self.role = {}
        self.alive = set(range(self.n))
        self.obs = {i: [] for i in range(self.n)}
        self.public = []

    def deal(self):
        cards = list(self.deal_override) if self.deal_override else self._shuffled()
        for i in range(self.n):
            self.role[i] = cards[i]
        # With more than one Mafia, the mafia recognize each other (a lone mafioso learns nothing new).
        mafia = self._mafia()
        if len(mafia) > 1:
            for m in mafia:
                others = [self.names[o] for o in mafia if o != m]
                self.obs[m].append(f"Your fellow Mafia: {', '.join(others)}.")

    def _shuffled(self):
        c = list(ROLES_BY_N[self.n])
        self.rng.shuffle(c)
        return c

    def _alive_list(self):
        return sorted(self.alive)

    def _mafia(self):
        return [i for i in self.alive if self.role[i] == "Mafia"]

    def _composition(self) -> str:
        """Human-readable role tally for the transcript header, e.g. '2 Mafia / 1 Doctor / ...'."""
        counts = Counter(self.role[i] for i in range(self.n))
        order = ["Mafia", "Doctor", "Detective", "Villager"]
        parts = [f"{counts[r]} {r}{'s' if counts[r] != 1 and r != 'Mafia' else ''}"
                 for r in order if counts.get(r)]
        return " / ".join(parts)

    def base_prompt(self, pid):
        lines = [
            f"You are {participant_label(self.names, pid)} in a {self.n}-player game of Secret Mafia.",
            f"Your role: {self.role[pid]}. {ROLE_DESC[self.role[pid]]}",
            f"Alive players: {', '.join(participant_label(self.names, i) for i in self._alive_list())}.",
            "When naming another participant in speech, use their @handle so references stay unambiguous.",
        ]
        if self.obs[pid]:
            lines.append("What you know privately:")
            lines += [f"  - {o}" for o in self.obs[pid]]
        if self.public:
            lines.append("Discussion and events so far:")
            lines += [f"  {p}" for p in self.public]
        return "\n".join(lines)

    def _pick(self, pid, agent, prompt_extra, candidates, default=None):
        prompt = self.base_prompt(pid) + prompt_extra

        def parse(a, raw):
            t = seat_for_participant_ref(a, self.names, candidates)
            if t not in candidates:
                raise ValueError("bad target")
            return t

        resp = agent.act(prompt, parse, default if default is not None else candidates[0])
        return resp.action, resp.declared_reasoning, resp.ms

    def run_night(self, rnd, agents):
        events = [{"t": "sys", "text": f"Night {rnd}: the town sleeps."}]
        reason = {}
        mafia = self._mafia()
        victims = [i for i in self.alive if i not in mafia]
        kill = None
        if mafia and victims:
            kill, r, ms = self._pick(mafia[0], agents[mafia[0]],
                                  f"\n\nNIGHT (Mafia): choose a player to kill.\nReply JSON "
                                  '{"reasoning":"...","action":"@participant"}.',
                                  victims)
            reason[mafia[0]] = r
            events.append({"t": "act", "pid": mafia[0], "text": "Mafia targets " + self.names[kill], "ms": ms})
        protect = None
        doc = next((i for i in self.alive if self.role[i] == "Doctor"), None)
        if doc is not None:
            protect, r, ms = self._pick(doc, agents[doc],
                                    "\n\nNIGHT (Doctor): choose a player to protect (may be yourself).\nReply JSON "
                                    '{"reasoning":"...","action":"@participant"}.',
                                    self._alive_list(), default=doc)
            reason[doc] = r
            events.append({"t": "act", "pid": doc, "text": "Doctor protects " + self.names[protect], "ms": ms})
        det = next((i for i in self.alive if self.role[i] == "Detective"), None)
        if det is not None:
            others = [i for i in self.alive if i != det]
            tgt, r, ms = self._pick(det, agents[det],
                                "\n\nNIGHT (Detective): choose a player to investigate.\nReply JSON "
                                '{"reasoning":"...","action":"@participant"}.',
                                others)
            reason[det] = r
            is_maf = self.role[tgt] == "Mafia"
            self.obs[det].append(f"You investigated {self.names[tgt]}: {'MAFIA' if is_maf else 'not mafia'}.")
            events.append({"t": "act", "pid": det, "text": "Detective investigates " + self.names[tgt], "ms": ms})

        if kill is not None and kill != protect:
            self.alive.discard(kill)
            events.append({"t": "sys", "text": f"{self.names[kill]} was killed in the night."})
            self.public.append(f"{self.names[kill]} was found dead at dawn.")
        else:
            events.append({"t": "sys", "text": "No one died last night."})
            self.public.append("No one died last night.")
        synth = {"state": f"Night {rnd} resolved.", "key": "Mafia kills; the Doctor may have saved the target.",
                 "note": "Detective gained private information."}
        return {"name": f"Night {rnd}", "kind": "night", "events": events, "reason": reason, "synth": synth}

    def run_day(self, rnd, agents):
        events = [{"t": "sys", "text": f"Day {rnd}: discussion, then a vote."}]
        reason = {}
        order = [i for i in self._alive_list()]
        self.rng.shuffle(order)
        for sub in range(self.discussion_rounds):  # max rounds; ends early when a round is all-passes
            events.append({"t": "round", "text": f"Round {sub+1}"})
            spoke = 0
            for pid in order:
                prompt = self.base_prompt(pid) + (
                    "\n\nSpeak to the town — accuse, defend, or share (or fake) information.\n"
                    'Reply JSON {"reasoning":"...","action":"<what you say>"} or {"action":"pass"}.')

                def parse(a, raw):
                    s = str(a).strip()
                    if not s:
                        raise ValueError("empty")
                    return s
                resp = agents[pid].act(prompt, parse, "pass")
                reason[pid] = resp.declared_reasoning
                if resp.action.lower() in ("pass", "(pass)"):
                    events.append({"t": "pass", "pid": pid, "ms": resp.ms})
                    self.public.append(f"{self.names[pid]} passes.")
                else:
                    events.append({"t": "say", "pid": pid, "text": resp.action, "ms": resp.ms})
                    self.public.append(f"{self.names[pid]}: {resp.action}")
                    spoke += 1
            if spoke == 0:
                events.append({"t": "sys", "text": "A full round passed in silence — discussion ends."})
                break
        # vote
        votes = {}
        vote_ms = {}
        for pid in order:
            targets = [i for i in self._alive_list() if i != pid]
            tgt, r, ms = self._pick(pid, agents[pid],
                                "\n\nVOTE: name the player to eliminate.\nReply JSON "
                                '{"reasoning":"...","action":"@participant"}.',
                                targets)
            votes[pid] = tgt; vote_ms[pid] = ms
        for pid in order:
            events.append({"t": "vote", "pid": pid, "tgt": votes[pid], "text": "votes", "ms": vote_ms[pid]})
        counts = {}
        for t in votes.values():
            counts[t] = counts.get(t, 0) + 1
        top = max(counts.values())
        leaders = [p for p, c in counts.items() if c == top]
        tally = sorted([{"pid": p, "n": c} for p, c in counts.items()], key=lambda x: -x["n"])
        eliminated = leaders[0] if len(leaders) == 1 else None
        if eliminated is not None:
            self.alive.discard(eliminated)
            events.append({"t": "sys", "text": f"{self.names[eliminated]} was voted out (was {self.role[eliminated]})."})
            self.public.append(f"{self.names[eliminated]} was voted out — they were a {self.role[eliminated]}.")
        else:
            events.append({"t": "sys", "text": "The vote tied — no one was eliminated."})
            self.public.append("The vote tied; no elimination.")
        synth = {"state": f"Day {rnd}: " + (f"{self.names[eliminated]} eliminated." if eliminated is not None else "no elimination."),
                 "key": "Town tries to vote out the Mafia on partial information.", "note": "Votes are public."}
        return {"name": f"Day {rnd}", "kind": "vote", "events": events, "reason": reason, "synth": synth,
                "tally": tally, "deaths": [eliminated] if eliminated is not None else []}

    def _winner(self):
        maf = len(self._mafia())
        if maf == 0:
            return "good"
        if maf >= len(self.alive) - maf:
            return "evil"
        return None

    def play(self, agents):
        self.deal()
        phases = []
        winner = None
        for rnd in range(1, MAX_ROUNDS + 1):
            phases.append(self.run_night(rnd, agents))
            winner = self._winner()
            if winner:
                break
            phases.append(self.run_day(rnd, agents))
            winner = self._winner()
            if winner:
                break
        if winner is None:
            winner = "evil"  # mafia survived to the round cap
        text = ("Village eliminated the Mafia — VILLAGE WINS." if winner == "good"
                else "Mafia reached parity — MAFIA WINS.")
        phases.append({"name": "Result", "kind": "result",
                       "events": [{"t": "result", "team": winner, "text": text}], "reason": {},
                       "synth": {"state": text, "key": "Win by elimination (village) or parity (mafia).",
                                 "note": "Per-seat reasoning retained for scoring."},
                       "outcome": {"team": winner, "text": text}})
        players = [{
            "seat": i, "dealt": self.role[i], "end": self.role[i], "team": team_of(self.role[i]),
            "believes": self.role[i], "won": (team_of(self.role[i]) == winner),
            **agent_stats(agents[i]),
        } for i in range(self.n)]
        return {
            "game": self.GAME, "title": self.TITLE, "seed": self.seed,
            "meta": f"{self.n} agents · {self._composition()} · seed {self.seed}",
            "players": players,
            "agentCallLog": agent_call_log(agents),
            "cardsInPlay": [[self.role[i], team_of(self.role[i])] for i in range(self.n)],
            "center": None, "phases": phases, "outcome": phases[-1]["outcome"], "winner_team": winner,
        }
