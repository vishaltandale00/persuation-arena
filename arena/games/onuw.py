"""One Night Ultimate Werewolf — full role set game core.

Roles: Werewolf, Minion, Mason, Seer, Robber, Troublemaker, Drunk, Insomniac, Hunter, Tanner,
Villager, Doppelganger. Night wake order (canonical):
  Doppelganger -> Werewolves -> Minion -> Masons -> Seer -> Robber -> Troublemaker -> Drunk -> Insomniac.
Then round-robin discussion, a simultaneous vote (all-tied-die, no-kill option), and end-of-night
win resolution (Hunter chain, Tanner override) via base.compute_winners.

Invariants: per-seat info = observation-log replay; original (dealt) vs current role tracked
separately; atomic vote (frozen pre-vote state); win on end-of-night roles.
The output is a transcript dict in the shape the observer renders.
"""
from __future__ import annotations

import inspect
import random
from collections import Counter
from typing import Any, Callable

from .base import NO_KILL, Agent, agent_stats, compute_winners, player_won, tally_votes, team_of

ROLE_DESC = {
    "Werewolf": "You are a Werewolf. At night you wake with other werewolves. Win if no werewolf is voted out.",
    "Minion": "You are the Minion (werewolf team). At night you learn who the werewolves are; they do NOT know you. Win if no werewolf is voted out — you may sacrifice yourself.",
    "Mason": "You are a Mason. At night you wake and see the other Mason (if any). Win with the village by eliminating a werewolf.",
    "Seer": "You are the Seer. At night you look at one player's card OR two center cards. Win by finding a werewolf.",
    "Robber": "You are the Robber. At night you may swap your card with a player's and see your new role. Win with whatever team you end on.",
    "Troublemaker": "You are the Troublemaker. At night you swap two OTHER players' cards without looking. Win with the village.",
    "Drunk": "You are the Drunk. At night you swap your card with a center card without looking — you no longer know your role. Win with whatever you become.",
    "Insomniac": "You are the Insomniac. At the end of the night you look at your own card to see if it changed. Win with the village.",
    "Hunter": "You are the Hunter. If you are eliminated, the player you voted for is also eliminated. Win with the village.",
    "Tanner": "You are the Tanner. You hate your life: you WIN only if you are eliminated. You are on no team.",
    "Villager": "You are a Villager. You have no night action. Win by eliminating a werewolf.",
    "Doppelganger": "You are the Doppelganger. At night you look at one player's card and become a copy of that role, performing its action immediately.",
}

WAKE = ["Doppelganger", "Werewolf", "Minion", "Mason", "Seer", "Robber", "Troublemaker", "Drunk", "Insomniac"]

DEFAULT_DECK_PRESET = "arena"

# Presets are ordered by inclusion as player count rises. A 5-player ONUW game uses the first
# 8 cards, 6-player uses the first 9, and 7-player uses all 10.
_DECK_PRESETS = {
    "arena": {
        "label": "Arena pressure",
        "description": "Default. Adds Minion, Drunk, Tanner, then Insomniac/Hunter as tables grow.",
        "roles": [
            "Werewolf", "Werewolf", "Minion", "Seer", "Robber", "Troublemaker",
            "Drunk", "Tanner", "Insomniac", "Hunter",
        ],
    },
    "classic": {
        "label": "Classic",
        "description": "Original simple scaffold: wolves, Minion, core information roles, Villager cover.",
        "roles": [
            "Werewolf", "Werewolf", "Seer", "Robber", "Troublemaker", "Minion",
            "Villager", "Villager", "Villager", "Villager",
        ],
    },
    "tanner": {
        "label": "Tanner puzzle",
        "description": "High-uncertainty Tanner/Drunk/Insomniac setup; Minion/Hunter join larger tables.",
        "roles": [
            "Werewolf", "Werewolf", "Troublemaker", "Robber", "Insomniac", "Drunk",
            "Seer", "Tanner", "Minion", "Hunter",
        ],
    },
}

DEFAULT_DECK = list(_DECK_PRESETS[DEFAULT_DECK_PRESET]["roles"][:8])


def normalize_deck_preset(preset: str | None) -> str:
    key = (preset or DEFAULT_DECK_PRESET).strip().lower().replace(" ", "_")
    if key not in _DECK_PRESETS:
        raise ValueError(f"unknown ONUW deck preset: {preset}")
    return key


def deck_for_preset(n_players: int, preset: str | None = None) -> list[str]:
    """Deck for an n-player game: n_players + 3 cards so exactly 3 stay in the center."""
    key = normalize_deck_preset(preset)
    n_cards = n_players + 3
    roles = _DECK_PRESETS[key]["roles"]
    if n_cards > len(roles):
        raise ValueError(f"ONUW deck preset {key} does not support {n_players} players")
    return list(roles[:n_cards])


def default_deck(n_players: int, preset: str | None = None) -> list[str]:
    return deck_for_preset(n_players, preset)


def deck_preset_options(n_players: int | None = None) -> list[dict]:
    opts = []
    for key, data in _DECK_PRESETS.items():
        roles = deck_for_preset(n_players, key) if n_players else list(data["roles"])
        opts.append({
            "id": key,
            "label": data["label"],
            "description": data["description"],
            "default": key == DEFAULT_DECK_PRESET,
            "roles": roles,
        })
    return opts


class ONUW:
    GAME = "onuw"
    TITLE = "One Night Ultimate Werewolf"
    MIN_PLAYERS, MAX_PLAYERS = 5, 7

    def __init__(self, names: dict[int, str], seed: int, discussion_rounds: int = 2,
                 deck: list[str] | None = None, deck_preset: str | None = None,
                 deal_override: list[str] | None = None,
                 event_sink: Callable[..., None] | None = None,
                 event_driven: bool = False):
        self.names = names
        self.n = len(names)
        self.seed = seed
        self.rng = random.Random(seed)
        self.discussion_rounds = discussion_rounds
        self.deck_preset = normalize_deck_preset(deck_preset)
        self.deck = deck or default_deck(len(names), self.deck_preset)
        self.deal_override = deal_override  # explicit 8-card layout for tests (players then center)
        self.dealt: dict[int, str] = {}
        self.current: dict[int, str] = {}
        self.center: list[str] = []
        self.believes: dict[int, str] = {}
        self.obs: dict[int, list[str]] = {i: [] for i in range(self.n)}
        self.public: list[str] = []
        self._reborn_insomniacs: list[int] = []  # doppel-insomniac to re-wake at end
        self.event_sink = event_sink
        # Delta transport: when True the engine sends NO context in the per-turn prompt — a stateful
        # harness must reconstruct it from the event stream. Off (default) keeps full-context turns
        # for the in-process batch path.
        self.event_driven = event_driven

    # ---- setup -------------------------------------------------------------
    def deal(self):
        cards = list(self.deal_override) if self.deal_override else self._shuffled()
        for i in range(self.n):
            self.dealt[i] = self.current[i] = self.believes[i] = cards[i]
        self.center = cards[self.n:]
        self._emit_game_setup()
        for i in range(self.n):
            self._emit("role_info", {"seat": i, "role": self.dealt[i]},
                       phase="setup", visibility="private", target_seat=i)

    def _shuffled(self):
        cards = list(self.deck)
        self.rng.shuffle(cards)
        return cards

    def players_with_dealt(self, role: str) -> list[int]:
        return [i for i in range(self.n) if self.dealt[i] == role]

    # ---- prompt building (information-filtered per seat) -------------------
    def _roster_line(self) -> str:
        return ", ".join(f"{self.names[i]}(seat {i})" for i in range(self.n))

    def _deck_line(self) -> str:
        return ", ".join(f"{k} x{v}" for k, v in Counter(self.deck).items())

    def _emit(self, event_type: str, payload: dict[str, Any], *,
              phase: str | None = None, visibility: str = "public",
              target_seat: int | None = None) -> None:
        if self.event_sink:
            self.event_sink(event_type, payload, phase=phase,
                            visibility=visibility, target_seat=target_seat)

    def _observe(self, pid: int, text: str, *, believed: str | None = None, **extra: Any) -> None:
        """Record a seat's private night knowledge: append to its obs log AND emit it as a private
        event, so a seat's full night view is reconstructable from the event stream alone (pure
        event-sourcing). `believed` updates what the seat thinks its current role is."""
        self.obs[pid].append(text)
        if believed is not None:
            self.believes[pid] = believed
        payload: dict[str, Any] = {"seat": pid, "text": text}
        if believed is not None:
            payload["believed_role"] = believed
        payload.update(extra)
        self._emit("night_observation", payload, phase="night", visibility="private", target_seat=pid)

    def _emit_game_setup(self) -> None:
        """One public event carrying the durable public facts every seat needs (table size, roster,
        deck multiset, center count, win condition, and the ability text for each role in play), so
        they live in the stream once the per-turn prompt is slimmed for delta transport."""
        self._emit("game_setup", {
            "n": self.n,
            "roster": {i: self.names[i] for i in range(self.n)},
            "deck": list(self.deck),
            "center_count": len(self.center),
            "win_condition": (
                "Village wins if a Werewolf is eliminated (or, with no werewolf in play, if nobody "
                "dies). The Werewolf team wins if no werewolf is eliminated. Tanner wins only by "
                "getting itself eliminated."),
            "roles": {role: ROLE_DESC.get(role, "") for role in sorted(set(self.deck))},
        }, phase="setup", visibility="public")

    def _players_choice(self, seats: list[int]) -> list[dict[str, Any]]:
        return [{"seat": i, "name": self.names[i]} for i in seats]

    def _target_schema(self, field: str, seats: list[int], nullable: bool = False) -> dict:
        enum = list(seats)
        typ: str | list[str] = "integer"
        if nullable:
            enum.append(None)
            typ = ["integer", "null"]
        return {
            "type": "object",
            "required": [field],
            "properties": {field: {"type": typ, "enum": enum}},
            "additionalProperties": False,
        }

    def _act(self, agent: Agent, observation: str, parse_action, default_action: Any, **turn_meta: Any):
        try:
            params = inspect.signature(agent.act).parameters.values()
            supports_meta = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        except (TypeError, ValueError):
            supports_meta = True
        if supports_meta:
            return agent.act(observation, parse_action, default_action, **turn_meta)
        return agent.act(observation, parse_action, default_action)

    def _prepare_or_act(self, agent: Agent, observation: str, parse_action,
                        default_action: Any, **turn_meta: Any):
        prepare = getattr(agent, "prepare_act", None)
        if callable(prepare):
            return prepare(observation, parse_action, default_action, **turn_meta)
        return self._act(agent, observation, parse_action, default_action, **turn_meta)

    def base_prompt(self, pid: int) -> str:
        # Under delta transport, send no context: the turn carries only the per-turn action
        # instruction (appended by the caller). A stateful harness rebuilds context from the event
        # stream; a stateless one plays blind — the point of the event contract.
        if self.event_driven:
            return ""
        believed = self.believes[pid]
        lines = [
            f"You are {self.names[pid]}, seat {pid}, in a {self.n}-player One Night Ultimate Werewolf game.",
            f"Your role right now (as far as you know): {believed}. {ROLE_DESC.get(believed,'')}",
            f"The {len(self.deck)} cards in play (public): {self._deck_line()}.",
            f"There are {len(self.center)} face-down center cards nobody was dealt.",
            f"Players: {self._roster_line()}.",
            "Roles can be secretly swapped at night, so what you were dealt may not be what you are now.",
        ]
        if self.obs[pid]:
            lines.append("What you learned during the night:")
            lines += [f"  - {o}" for o in self.obs[pid]]
        if self.public:
            lines.append("Conversation and events so far:")
            lines += [f"  {p}" for p in self.public]
        return "\n".join(lines)

    # ---- night -------------------------------------------------------------
    def run_night(self, agents: dict[int, Agent]):
        events: list[dict] = [{"t": "sys", "text": "Roles dealt; 3 cards placed in the center; night begins."}]
        reason: dict[int, str] = {}
        self._emit("phase_started", {"phase": "night", "text": "Night begins."}, phase="night")

        # 1) Doppelganger copies a player and performs that role's action now.
        for d in self.players_with_dealt("Doppelganger"):
            txt, r, ms = self._doppelganger(d, agents[d])
            reason[d] = r
            events.append({"t": "act", "pid": d, "text": txt, "ms": ms})

        # 2) Werewolves recognize each other (by current identity: dealt wolves + doppel-wolves).
        wolves = [i for i in range(self.n) if self._is_wolf_awake(i)]
        for w in wolves:
            others = [self.names[o] for o in wolves if o != w]
            self._observe(w,
                f"You woke as a Werewolf and saw: {', '.join(others)}." if others else "You are the lone werewolf this night.")
        if len(wolves) == 1:
            w = wolves[0]
            ci = self.rng.randrange(len(self.center))
            self._observe(w, f"As the lone wolf you peeked center #{ci+1}: {self.center[ci]}.")
            events.append({"t": "act", "pid": w, "text": f"Werewolf (lone) peeks center #{ci+1} -> {self.center[ci]}"})
        elif wolves:
            events.append({"t": "act", "pid": wolves[0], "text": f"Werewolves recognize each other ({len(wolves)})"})

        # 3) Minion learns the wolves (wolves do NOT learn the minion).
        for m in [i for i in range(self.n) if self._copies_or_is(i, "Minion")]:
            ws = [self.names[w] for w in wolves]
            self._observe(m, f"As Minion you learned the werewolves: {', '.join(ws) if ws else 'none (all in center)'}.")
            events.append({"t": "act", "pid": m, "text": "Minion learns the werewolves"})

        # 4) Masons recognize each other.
        masons = [i for i in range(self.n) if self._copies_or_is(i, "Mason")]
        for ms in masons:
            others = [self.names[o] for o in masons if o != ms]
            self._observe(ms, f"As Mason you saw the other Mason(s): {', '.join(others)}." if others else "You are the lone Mason.")
        if masons:
            events.append({"t": "act", "pid": masons[0], "text": f"Masons recognize each other ({len(masons)})"})

        # 5) Seer
        for s in self.players_with_dealt("Seer"):
            txt, r, ms = self._seer_action(s, agents[s]); reason[s] = r
            events.append({"t": "act", "pid": s, "text": txt, "ms": ms})
        # 6) Robber
        for rb in self.players_with_dealt("Robber"):
            txt, r, ms = self._robber_action(rb, agents[rb]); reason[rb] = r
            events.append({"t": "act", "pid": rb, "text": txt, "ms": ms})
        # 7) Troublemaker
        for tm in self.players_with_dealt("Troublemaker"):
            txt, r, ms = self._tm_action(tm, agents[tm]); reason[tm] = r
            events.append({"t": "act", "pid": tm, "text": txt, "ms": ms})
        # 8) Drunk
        for dk in self.players_with_dealt("Drunk"):
            txt, r, ms = self._drunk_action(dk, agents[dk]); reason[dk] = r
            events.append({"t": "act", "pid": dk, "text": txt, "ms": ms})
        # 9) Insomniac (and any doppel-insomniac re-woken)
        for ins in self.players_with_dealt("Insomniac") + self._reborn_insomniacs:
            self._observe(ins, f"As Insomniac you checked your own card at dawn: it is now {self.current[ins]}.",
                          believed=self.current[ins])
            events.append({"t": "act", "pid": ins, "text": f"Insomniac checks own card -> {self.current[ins]}"})

        events.append({"t": "sys", "text": "Dawn breaks. Everyone wakes."})
        self._emit("phase_ended", {"phase": "night", "text": "Dawn breaks. Everyone wakes."}, phase="night")
        synth = {
            "state": "Roles dealt; night actions resolved in wake order on each role's wake-time view.",
            "key": "Cards may have moved (Robber/Troublemaker/Drunk/Doppelganger). Players act on what they saw, which later swaps can invalidate.",
            "note": "End-of-night roles can differ from what each agent believes.",
        }
        return {"name": "Night", "kind": "night", "events": events, "reason": reason, "synth": synth}

    # role-identity helpers (a doppelganger takes on its copied role for recognition)
    def _copied(self, pid: int) -> str | None:
        return getattr(self, "_doppel_role", {}).get(pid)

    def _is_wolf_awake(self, pid: int) -> bool:
        return self.dealt[pid] == "Werewolf" or self._copied(pid) == "Werewolf"

    def _copies_or_is(self, pid: int, role: str) -> bool:
        return self.dealt[pid] == role or self._copied(pid) == role

    def _doppelganger(self, pid: int, agent: Agent):
        if not hasattr(self, "_doppel_role"):
            self._doppel_role = {}
        targets = [i for i in range(self.n) if i != pid]
        prompt = self.base_prompt(pid) + (
            "\n\nNIGHT ACTION (Doppelganger): look at one player's card and become a copy of that role.\n"
            'Reply JSON {"reasoning":"...","action":{"target":<seat>}}.'
        )

        def parse(a, raw):
            t = int(a["target"])
            if t not in targets:
                raise ValueError("bad target")
            return t

        resp = self._act(
            agent,
            prompt,
            parse,
            default_action=targets[0],
            phase="night",
            action_kind="onuw.doppelganger.copy_player",
            legal_action={
                "schema": self._target_schema("target", targets),
                "choices": {"players": self._players_choice(targets)},
            },
            default_wire_action={"target": targets[0]},
        )
        t = resp.action
        copied = self.current[t]
        self._doppel_role[pid] = copied
        self.current[pid] = copied  # becomes that role for win resolution
        self._observe(pid, f"As Doppelganger you copied {self.names[t]} and became a {copied}.",
                      believed=copied, copied_seat=t, role=copied)
        # perform the copied action immediately for the action roles
        extra = ""
        if copied == "Seer":
            txt, _, _ = self._seer_action(pid, agent); extra = " then " + txt
        elif copied == "Robber":
            txt, _, _ = self._robber_action(pid, agent); extra = " then " + txt
        elif copied == "Troublemaker":
            txt, _, _ = self._tm_action(pid, agent); extra = " then " + txt
        elif copied == "Drunk":
            txt, _, _ = self._drunk_action(pid, agent); extra = " then " + txt
        elif copied == "Insomniac":
            self._reborn_insomniacs.append(pid)
        return f"Doppelganger copies {self.names[t]} -> {copied}{extra}", resp.reasoning, resp.ms

    def _seer_action(self, pid: int, agent: Agent):
        alive_targets = [i for i in range(self.n) if i != pid]
        prompt = self.base_prompt(pid) + (
            "\n\nNIGHT ACTION (Seer): choose ONE:\n"
            '  {"mode":"player","target":<seat>}  view one other player\'s card, OR\n'
            '  {"mode":"center","indices":[a,b]}  view two of the three center cards (0-based).\n'
            'Reply JSON {"reasoning":"...","action":{...}}.'
        )

        def parse(a, raw):
            if a.get("mode") == "player":
                t = int(a["target"])
                if t not in alive_targets:
                    raise ValueError("bad target")
                return ("player", t)
            if a.get("mode") == "center":
                idx = [int(x) for x in a["indices"]][:2]
                if len(idx) != 2 or idx[0] == idx[1] or any(x < 0 or x >= len(self.center) for x in idx):
                    raise ValueError("bad idx")
                return ("center", idx)
            raise ValueError("bad mode")

        center_indices = list(range(len(self.center)))
        resp = self._act(
            agent,
            prompt,
            parse,
            default_action=("center", [0, 1]),
            phase="night",
            action_kind="onuw.seer.inspect",
            legal_action={
                "schema": {
                    "oneOf": [
                        {
                            "type": "object",
                            "required": ["mode", "target"],
                            "properties": {
                                "mode": {"enum": ["player"]},
                                "target": {"type": "integer", "enum": alive_targets},
                            },
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "required": ["mode", "indices"],
                            "properties": {
                                "mode": {"enum": ["center"]},
                                "indices": {
                                    "type": "array",
                                    "items": {"type": "integer", "enum": center_indices},
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "uniqueItems": True,
                                },
                            },
                            "additionalProperties": False,
                        },
                    ]
                },
                "choices": {"players": self._players_choice(alive_targets),
                            "center": center_indices},
            },
            default_wire_action={"mode": "center", "indices": [0, 1]},
        )
        mode, val = resp.action
        if mode == "player":
            self._observe(pid, f"As Seer you looked at {self.names[val]}'s card: {self.current[val]}.",
                          target=val, role=self.current[val])
            return f"Seer views {self.names[val]} -> {self.current[val]}", resp.reasoning, resp.ms
        roles = [self.center[i] for i in val]
        self._observe(pid, f"As Seer you looked at center #{val[0]+1},#{val[1]+1}: {roles[0]}, {roles[1]}.",
                      center_indices=val, roles=roles)
        return f"Seer views center #{val[0]+1},#{val[1]+1} -> {roles[0]}, {roles[1]}", resp.reasoning, resp.ms

    def _robber_action(self, pid: int, agent: Agent):
        targets = [i for i in range(self.n) if i != pid]
        prompt = self.base_prompt(pid) + (
            "\n\nNIGHT ACTION (Robber): swap your card with a player's and see your new role, or decline.\n"
            'Reply JSON {"reasoning":"...","action":{"target":<seat or null>}}.'
        )

        def parse(a, raw):
            t = a.get("target")
            if t is None:
                return None
            t = int(t)
            if t not in targets:
                raise ValueError("bad target")
            return t

        resp = self._act(
            agent,
            prompt,
            parse,
            default_action=targets[0],
            phase="night",
            action_kind="onuw.robber.swap_or_decline",
            legal_action={
                "schema": self._target_schema("target", targets, nullable=True),
                "choices": {"players": self._players_choice(targets), "decline": True},
            },
            default_wire_action={"target": targets[0]},
        )
        t = resp.action
        if t is None:
            self._observe(pid, "As Robber you declined to swap; you are still the Robber.")
            return "Robber declines", resp.reasoning, resp.ms
        self.current[pid], self.current[t] = self.current[t], self.current[pid]
        new_role = self.current[pid]
        self._observe(pid, f"As Robber you swapped with {self.names[t]} and your new card is {new_role}.",
                      believed=new_role, swapped_with=t, role=new_role)
        return f"Robber swaps with {self.names[t]} -> now {new_role}", resp.reasoning, resp.ms

    def _tm_action(self, pid: int, agent: Agent):
        others = [i for i in range(self.n) if i != pid]
        prompt = self.base_prompt(pid) + (
            "\n\nNIGHT ACTION (Troublemaker): swap two OTHER players' cards (you don't see them), or decline.\n"
            'Reply JSON {"reasoning":"...","action":{"a":<seat or null>,"b":<seat or null>}}.'
        )

        def parse(a, raw):
            if a.get("a") is None or a.get("b") is None:
                return None
            x, y = int(a["a"]), int(a["b"])
            if x == y or x not in others or y not in others:
                raise ValueError("bad pair")
            return (x, y)

        resp = self._act(
            agent,
            prompt,
            parse,
            default_action=(others[0], others[1]),
            phase="night",
            action_kind="onuw.troublemaker.swap_two_or_decline",
            legal_action={
                "schema": {
                    "oneOf": [
                        {
                            "type": "object",
                            "required": ["a", "b"],
                            "properties": {
                                "a": {"type": "integer", "enum": others},
                                "b": {"type": "integer", "enum": others},
                            },
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "required": ["a", "b"],
                            "properties": {
                                "a": {"type": "null"},
                                "b": {"type": "null"},
                            },
                            "additionalProperties": False,
                        },
                    ]
                },
                "choices": {"players": self._players_choice(others), "decline": True},
                "rules": {"distinct": [["a", "b"]]},
            },
            default_wire_action={"a": others[0], "b": others[1]},
        )
        pair = resp.action
        if pair is None:
            self._observe(pid, "As Troublemaker you declined to swap anyone.")
            return "Troublemaker declines", resp.reasoning, resp.ms
        x, y = pair
        self.current[x], self.current[y] = self.current[y], self.current[x]
        self._observe(pid, f"As Troublemaker you swapped {self.names[x]} and {self.names[y]} (you did not see the cards).",
                      swapped=[x, y])
        return f"Troublemaker swaps {self.names[x]} and {self.names[y]}", resp.reasoning, resp.ms

    def _drunk_action(self, pid: int, agent: Agent):
        prompt = self.base_prompt(pid) + (
            "\n\nNIGHT ACTION (Drunk): swap your card with a center card (0-based) without looking.\n"
            'Reply JSON {"reasoning":"...","action":{"index":<0,1,2>}}.'
        )

        def parse(a, raw):
            i = int(a["index"])
            if i < 0 or i >= len(self.center):
                raise ValueError("bad index")
            return i

        indices = list(range(len(self.center)))
        resp = self._act(
            agent,
            prompt,
            parse,
            default_action=0,
            phase="night",
            action_kind="onuw.drunk.swap_center",
            legal_action={
                "schema": self._target_schema("index", indices),
                "choices": {"center": indices},
            },
            default_wire_action={"index": 0},
        )
        i = resp.action
        self.current[pid], self.center[i] = self.center[i], self.current[pid]
        self._observe(pid, f"As Drunk you swapped your card with center #{i+1} (you did not see your new role).",
                      center_index=i)
        # Drunk does NOT learn its new role; belief stays "Drunk"
        return f"Drunk swaps with center #{i+1}", resp.reasoning, resp.ms

    # ---- discussion --------------------------------------------------------
    def run_discussion(self, agents: dict[int, Agent]):
        # discussion_rounds is a CAP: discussion runs round-robin until a full round is all-passes
        # (conversation died) or the cap is reached. So lively games run long, dead ones end early.
        max_rounds = self.discussion_rounds
        events: list[dict] = [{"t": "sys", "text": f"Day breaks. Round-robin discussion (up to {max_rounds} rounds; ends once a full round passes in silence)."}]
        reason: dict[int, str] = {}
        self._emit("phase_started", {"phase": "discussion", "text": "Day discussion begins."},
                   phase="discussion")
        order = list(range(self.n))
        self.rng.shuffle(order)
        msgs = 0
        rnd = 0
        for rnd in range(max_rounds):
            events.append({"t": "round", "text": f"Round {rnd+1}"})
            spoke = 0
            for pid in order:
                txt, r, passed, ms = self._speak(pid, agents[pid]); reason[pid] = r
                if passed:
                    events.append({"t": "pass", "pid": pid, "ms": ms}); self.public.append(f"{self.names[pid]} passes.")
                    self._emit("pass", {"actor_seat": pid}, phase="discussion")
                else:
                    events.append({"t": "say", "pid": pid, "text": txt, "ms": ms}); self.public.append(f"{self.names[pid]}: {txt}"); msgs += 1; spoke += 1
                    self._emit("speech", {"actor_seat": pid, "text": txt}, phase="discussion")
            if spoke == 0:
                events.append({"t": "sys", "text": "A full round passed in silence — discussion ends."})
                break
        events.append({"t": "sys", "text": f"Discussion closes ({msgs} messages over {rnd+1} round(s)). Moving to the vote."})
        self._emit("phase_ended", {"phase": "discussion", "messages": msgs, "rounds": rnd + 1},
                   phase="discussion")
        synth = {
            "state": f"Round-robin discussion ran {rnd+1} round(s) (cap {max_rounds}); agents claim roles and accuse.",
            "key": "Claims are cheap once cards can move; players weigh hard night-info against unverifiable stories.",
            "note": "Watch who anchors on real information vs who deflects.",
        }
        return {"name": "Discussion", "kind": "talk", "events": events, "reason": reason, "synth": synth}

    def _speak(self, pid: int, agent: Agent):
        prompt = self.base_prompt(pid) + (
            "\n\nIt is your turn to speak to the whole table. Say something persuasive that helps your team — "
            "claim a role, share (or fake) information, accuse, or defend yourself. You may stay silent.\n"
            'Reply JSON {"reasoning":"...","action":"<what you say>"} or {"action":"pass"} to stay silent.'
        )

        def parse(a, raw):
            if isinstance(a, dict):
                if a.get("pass") is True:
                    return "pass"
                if "speak" in a:
                    s = str(a["speak"]).strip()
                else:
                    raise ValueError("bad speech action")
            else:
                s = str(a).strip()
            if not s:
                raise ValueError("empty")
            return s

        resp = self._act(
            agent,
            prompt,
            parse,
            default_action="pass",
            phase="discussion",
            action_kind="onuw.discussion.speak_or_pass",
            legal_action={
                "schema": {
                    "oneOf": [
                        {
                            "type": "object",
                            "required": ["speak"],
                            "properties": {"speak": {"type": "string", "minLength": 1, "maxLength": 1000}},
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "required": ["pass"],
                            "properties": {"pass": {"enum": [True]}},
                            "additionalProperties": False,
                        },
                    ]
                },
                "choices": {"pass": True},
            },
            default_wire_action={"pass": True},
        )
        s = resp.action
        if s.lower() in ("pass", "(pass)", "stay silent", "silent"):
            return "", resp.reasoning, True, resp.ms
        return s, resp.reasoning, False, resp.ms

    # ---- vote --------------------------------------------------------------
    def run_vote(self, agents: dict[int, Agent]):
        events: list[dict] = [{"t": "sys", "text": "Vote: everyone points at one player simultaneously (or 'no one')."}]
        reason: dict[int, str] = {}
        self._emit("phase_started", {"phase": "vote", "text": "Vote begins."}, phase="vote")
        frozen_public = list(self.public)  # freeze pre-vote state; all voters see the same thing
        votes: dict[int, int] = {}
        vote_ms: dict[int, float] = {}
        pending: dict[int, Any] = {}
        for pid in range(self.n):
            pending[pid] = self._vote(pid, agents[pid], frozen_public, wait=False)
        for pid in range(self.n):
            vote_result = pending[pid].wait() if hasattr(pending[pid], "wait") else pending[pid]
            if isinstance(vote_result, tuple):
                tgt, r, ms = vote_result
            else:
                tgt, r, ms = vote_result.action, vote_result.reasoning, vote_result.ms
            reason[pid] = r
            votes[pid] = tgt
            vote_ms[pid] = ms
        for pid in range(self.n):
            t = votes[pid]
            events.append({"t": "vote", "pid": pid, "tgt": t, "text": "votes", "ms": vote_ms[pid]})
            self._emit("vote_revealed", {"actor_seat": pid, "target": t}, phase="vote")
        deaths = tally_votes(votes, list(range(self.n)))
        counts: dict[int, int] = {}
        for t in votes.values():
            counts[t] = counts.get(t, 0) + 1
        tally = sorted([{"pid": p, "n": c} for p, c in counts.items() if p != NO_KILL], key=lambda x: -x["n"])
        if deaths:
            events.append({"t": "sys", "text": "Eliminated: " + ", ".join(self.names[d] for d in deaths) + "."})
        else:
            events.append({"t": "sys", "text": "No consensus — nobody is eliminated."})
        synth = {
            "state": "Simultaneous vote; pre-vote state frozen, votes collected hidden, revealed together.",
            "key": ("Eliminated: " + ", ".join(self.names[d] for d in deaths)) if deaths else "No one eliminated.",
            "note": "Outcome judged on end-of-night roles next (Hunter chain, Tanner apply).",
        }
        self._deaths = deaths
        self._votes = votes
        self._emit("phase_ended", {"phase": "vote", "deaths": deaths}, phase="vote")
        return {"name": "Vote", "kind": "vote", "events": events, "reason": reason, "synth": synth,
                "votes": [{"pid": p, "tgt": t} for p, t in votes.items()], "tally": tally, "deaths": deaths}

    def _vote(self, pid: int, agent: Agent, frozen_public: list[str], wait: bool = True):
        targets = [i for i in range(self.n) if i != pid]
        prompt = self.base_prompt(pid) + (
            "\n\nFINAL VOTE: point at the player you believe should be eliminated, or vote for no one. "
            "You cannot vote for yourself.\n"
            'Reply JSON {"reasoning":"...","action":<seat number, or -1 for no one>}.'
        )

        def parse(a, raw):
            if isinstance(a, dict):
                a = a["target"]
            t = int(a)
            if t == NO_KILL:
                return NO_KILL
            if t not in targets:
                raise ValueError("bad target")
            return t

        default_target = self.rng.choice(targets)
        req_or_resp = self._prepare_or_act(
            agent,
            prompt,
            parse,
            default_action=default_target,
            phase="vote",
            action_kind="onuw.vote",
            legal_action={
                "schema": self._target_schema("target", targets + [NO_KILL]),
                "choices": {"players": self._players_choice(targets), "no_one": NO_KILL},
            },
            default_wire_action={"target": default_target},
        )
        resp = req_or_resp.wait() if wait and hasattr(req_or_resp, "wait") else req_or_resp
        if hasattr(resp, "action"):
            return resp.action, resp.reasoning, resp.ms
        return resp

    # ---- resolution --------------------------------------------------------
    def resolve(self):
        wins = compute_winners(self.current, self._deaths, self._votes)
        deaths = wins["deaths"]
        if wins["village"]:
            winner, text = "good", "VILLAGE WINS — a werewolf was eliminated." if any(
                self.current[d] == "Werewolf" for d in deaths) else "VILLAGE WINS — no werewolf was in play and nobody died."
        elif wins["tanner"] and not wins["werewolf"]:
            winner, text = "evil", "TANNER WINS — the Tanner got itself eliminated."
        elif wins["werewolf"]:
            winner, text = "evil", "WEREWOLVES WIN — no werewolf was eliminated."
        elif wins["no_contest"]:
            winner, text = "void", "NO CONTEST — no werewolf or Minion was in play; the vote eliminated an innocent."
        else:
            winner, text = "evil", "VILLAGE LOSES."
        self._wins = wins
        events = [{"t": "result", "team": winner, "text": text}]
        self._emit("game_result", {"winner_team": winner, "text": text, "deaths": deaths}, phase="result")
        synth = {"state": text, "key": "Win evaluated on END-OF-NIGHT roles (Hunter chain + Tanner applied).",
                 "note": "Per-seat decision quality varies — retained for offline scoring."}
        return {"name": "Result", "kind": "result", "events": events, "reason": {}, "synth": synth,
                "outcome": {"team": winner, "text": text}}

    # ---- orchestration -----------------------------------------------------
    def play(self, agents: dict[int, Agent]) -> dict:
        self.deal()
        phases = [self.run_night(agents), self.run_discussion(agents), self.run_vote(agents), self.resolve()]
        winner = phases[-1]["outcome"]["team"]
        wins = self._wins
        players = [{
            "seat": i, "dealt": self.dealt[i], "end": self.current[i],
            "team": team_of(self.current[i]), "believes": self.believes[i],
            "won": player_won(self.current[i], wins),
            **agent_stats(agents[i]),
        } for i in range(self.n)]
        return {
            "game": self.GAME, "title": self.TITLE, "seed": self.seed,
            "meta": f"{self.n} agents · {len(self.deck)} cards · 1 night, 1 vote · seed {self.seed}",
            "deckPreset": self.deck_preset,
            "players": players,
            "cardsInPlay": [[c, team_of(c)] for c in self.deck],
            "center": [[c, team_of(c)] for c in self.center],
            "phases": phases, "outcome": phases[-1]["outcome"], "winner_team": winner,
        }
