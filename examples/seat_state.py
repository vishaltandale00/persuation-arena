"""Reference reducer for a STATEFUL ONUW harness — rebuilds a seat's view purely from the event
stream (pure event-sourcing).

This is *harness* code, not part of the Arena SDK. The SDK only delivers the raw delta event
stream; whether a harness accumulates it into state (and can therefore play well) is entirely the
harness's choice. Under delta transport a stateless harness sees only new events and goes blind —
which is exactly what the event contract is meant to select against.

Usage in a harness:
    st = SeatState(seat)
    # on each polled event:
    st.apply(event)
    # when it's your turn:
    prompt = st.render_context()   # == the engine's old base_prompt(seat), rebuilt from events

`apply` accepts either an SDK Event (with .type/.payload) or a plain dict {type, payload, ...}.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from arena.identity import participant_label, public_ref, roster_line
from arena.games.onuw import format_onuw_rules_block


def _etype(event: Any) -> str:
    return event["type"] if isinstance(event, dict) else event.type


def _payload(event: Any) -> dict:
    return (event["payload"] if isinstance(event, dict) else event.payload) or {}


@dataclass
class SeatState:
    """A seat's filtered view of an ONUW game, reconstructed by folding its event stream.

    `seat` may be left None: a live agent doesn't know which seat it's in until the first private
    event arrives (the server only delivers a seat its own private events), so we learn it then. The
    conformance test constructs SeatState(seat) with it already known."""
    seat: int | None = None
    n: int = 0
    roster: dict[int, str] = field(default_factory=dict)   # seat -> name
    deck: list[str] = field(default_factory=list)          # full multiset (players + center), public
    center_count: int = 0
    win_condition: str = ""
    roles: dict[str, str] = field(default_factory=dict)    # role -> ability text (for roles in play)
    believed_role: str | None = None                       # what this seat currently thinks it is
    night_obs: list[str] = field(default_factory=list)     # private night knowledge, in order
    public: list[str] = field(default_factory=list)        # public conversation ("name: text" / "name passes.")
    phase: str | None = None
    finished: bool = False
    result: dict | None = None

    def apply(self, event: Any) -> None:
        """Fold one event into the state. Safe to call on every polled event — events not visible
        to this seat (other seats' private events) never arrive, and unrelated types are ignored."""
        etype, p = _etype(event), _payload(event)
        if etype == "game_setup":
            self.n = int(p.get("n", self.n))
            if p.get("participants"):
                self.roster = {i: x.get("name", f"Participant {i + 1}")
                               for i, x in enumerate(p.get("participants") or [])}
            else:
                self.roster = {int(k): v for k, v in (p.get("roster") or {}).items()}
            self.deck = list(p.get("deck") or [])
            self.center_count = int(p.get("center_count", 0))
            self.win_condition = p.get("win_condition", "")
            self.roles = dict(p.get("roles") or {})
        elif etype == "role_info":            # the seat's dealt role, emitted once at setup
            self._learn_seat(p)
            if p.get("role") and self.believed_role is None:
                self.believed_role = p["role"]
        elif etype == "night_observation":    # this seat's private night knowledge
            self._learn_seat(p)
            self.night_obs.append(p.get("text", ""))
            if p.get("believed_role"):
                self.believed_role = p["believed_role"]
        elif etype == "speech":
            self.public.append(f"{self._actor_name(p)}: {p.get('text', '')}")
        elif etype == "pass":
            if p.get("stance") == "done":
                self.public.append(f"{self._actor_name(p)} is ready to end discussion.")
            elif p.get("stance") == "wait":
                self.public.append(f"{self._actor_name(p)} passes for now, waiting for more discussion.")
            else:
                self.public.append(f"{self._actor_name(p)} passes.")
        elif etype == "discussion_notice":
            self.public.append(p.get("text", ""))
        elif etype in ("phase_started", "phase_ended"):
            self.phase = p.get("phase", self.phase)
        elif etype == "game_result":
            self.finished = True
            self.result = p

    def _learn_seat(self, p: dict) -> None:
        # Private events delivered to this agent are always its own — learn our seat from the first.
        if self.seat is None and p.get("seat") is not None:
            self.seat = int(p["seat"])
        if self.seat is None and p.get("participant") and self.roster:
            ref = p["participant"].get("ref")
            for seat, name in self.roster.items():
                if public_ref(name) == ref:
                    self.seat = seat
                    break

    # --- views ---------------------------------------------------------------
    def _actor_name(self, payload: dict) -> str:
        actor = payload.get("actor")
        if isinstance(actor, dict) and actor.get("name"):
            return actor["name"]
        return self._name(payload.get("actor_seat"))

    def _name(self, seat: Any) -> str:
        try:
            return self.roster.get(int(seat), f"Participant {int(seat) + 1}")
        except (TypeError, ValueError):
            return "Participant"

    def deck_line(self) -> str:
        return ", ".join(f"{k} x{v}" for k, v in Counter(self.deck).items())

    def roster_line(self) -> str:
        return roster_line(self.roster)

    def render_context(self, *, phase: str = "context", action_kind: str = "onuw.context") -> str:
        """Reconstruct the engine's per-seat filtered prompt from the folded events.

        Intentionally byte-identical to the engine's ONUW.base_prompt(seat) so an agent driven by
        the delta event stream sees exactly what the old full-context turn used to hand it."""
        believed = self.believed_role
        me = participant_label(self.roster, self.seat) if self.seat is not None else "Participant (@participant)"
        lines = [
            f"You are {me} in a {self.n}-player One Night Ultimate Werewolf game.",
            f"Your role right now (as far as you know): {believed}. {self.roles.get(believed, '')}",
            f"The {len(self.deck)} cards in play (public): {self.deck_line()}.",
            f"There are {self.center_count} face-down center cards nobody was dealt.",
            f"Players: {self.roster_line()}.",
            "Roles can be secretly swapped at night, so what you were dealt may not be what you are now.",
            "When naming another participant in speech, use their @handle so references stay unambiguous.",
            "",
            format_onuw_rules_block(self.n, self.deck, self.center_count, phase=phase, action_kind=action_kind),
        ]
        if self.night_obs:
            lines.append("What you learned during the night:")
            lines += [f"  - {o}" for o in self.night_obs]
        if self.public:
            lines.append("Conversation and events so far:")
            lines += [f"  {p}" for p in self.public]
        return "\n".join(lines)
