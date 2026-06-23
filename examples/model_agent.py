"""Connected sample agent backed by a real LLM (via OpenRouter), STATEFUL over the delta stream.

The connected engine sends each seat only *new* events (deltas). This agent folds them into a
per-game SeatState (persuasion_arena_agent.state) via `on_event`, reconstructs the seat's filtered
view, and decides on its turn. It does NOT read the turn's observation for context — context comes
from the reconstructed event stream, and the action instruction is derived from the turn's
action_kind + legal_action. So it plays correctly even once the wire turn is slimmed to just the
action request (pure event-sourcing).

Output contract (the safety-net floor): prompt the model for JSON {reasoning, action}; extract and
validate against the turn's legal_action; on an illegal/missing move do ONE repair re-prompt that
feeds back the error, then fall back to a safe *legal* action so the seat never forfeits on bad
output. The server validates authoritatively regardless.

Per-game private notebook: the engine never feeds an agent its own prior reasoning back, so we keep
our own notes per game and re-supply them, to sustain multi-turn strategy/deception.

Wire it onto an ArenaAgent (the SDK owns polling + cursor; this owns game state):
    ma = ModelAgent("openai/gpt-5.5")
    agent.on_event(ma.on_event)
    agent.act(ma.act)
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from examples.seat_state import SeatState

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"

SYSTEM = (
    "You are a sharp, competitive player of a hidden-role social-deduction game. "
    "Read the situation, reason about who is lying and what serves your team, then act. "
    "Always reply with a single JSON object and nothing else: "
    '{"reasoning": "<your private thinking, never shown to others>", "action": <the action>}. '
    "Keep reasoning to a few sentences. Follow the action format exactly."
)

# Concise per-action instruction. Context comes from the reconstructed event view (not the turn);
# this is just "what move is being asked of you, and its shape".
ACTION_INSTRUCTIONS = {
    "onuw.discussion.speak_or_pass":
        'It is your turn to speak to the whole table. Be persuasive for YOUR team — claim a role, '
        'share or fake information, accuse, or defend yourself. '
        'action = {"speak": "<what you say>"}  or  {"pass": true} to stay silent.',
    "onuw.vote":
        'Vote for who should be eliminated. action = {"target": <seat>}  or  {"target": -1} for no one.',
    "onuw.seer.inspect":
        'NIGHT (Seer): gather information. action = {"mode":"player","target":<seat>} to see one '
        'player\'s card, OR {"mode":"center","indices":[a,b]} to see two center cards.',
    "onuw.robber.swap_or_decline":
        'NIGHT (Robber): action = {"target":<seat>} to swap your card with that player and learn '
        'your new role, OR {"target": null} to decline.',
    "onuw.troublemaker.swap_two_or_decline":
        'NIGHT (Troublemaker): action = {"a":<seat>,"b":<seat>} to swap two OTHER players\' cards '
        '(you will not see them), OR {"a": null, "b": null} to decline.',
    "onuw.drunk.swap_center":
        'NIGHT (Drunk): action = {"index": 0|1|2} to blindly swap your card with that center card.',
    "onuw.doppelganger.copy_player":
        'NIGHT (Doppelganger): action = {"target":<seat>} to look at that player\'s card and become '
        'a copy of that role.',
}


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        frag = m.group(0)
        for end in range(len(frag), 0, -1):
            try:
                return json.loads(frag[:end])
            except json.JSONDecodeError:
                continue
    return None


def _legal_players(turn) -> list[int]:
    return [p["seat"] for p in (turn.legal_action.get("choices", {}).get("players") or [])]


def _fallback_action(turn) -> dict:
    """A safe, legal action for each ONUW action kind — mirrors examples/pass_agent.py."""
    kind, players = turn.action_kind, _legal_players(turn)
    if kind == "onuw.discussion.speak_or_pass":
        return {"pass": True}
    if kind == "onuw.vote":
        return {"target": -1}
    if kind == "onuw.seer.inspect":
        return {"mode": "center", "indices": [0, 1]}
    if kind == "onuw.troublemaker.swap_two_or_decline":
        return {"a": None, "b": None}
    if kind == "onuw.robber.swap_or_decline":
        return {"target": players[0] if players else None}
    if kind == "onuw.doppelganger.copy_player":
        return {"target": players[0] if players else 0}
    if kind == "onuw.drunk.swap_center":
        return {"index": 0}
    return {"pass": True}


def _coerce(turn, a: Any) -> Any:
    """Forgive common shape slips before validating (e.g. a bare speech string)."""
    if turn.action_kind == "onuw.discussion.speak_or_pass" and isinstance(a, str):
        return {"pass": True} if a.strip().lower() in ("pass", "(pass)", "") else {"speak": a}
    return a


def _is_legal(turn, a: Any) -> bool:
    """Cheap pre-flight so an illegal model move becomes a fallback, not a forfeit."""
    if not isinstance(a, dict):
        return False
    kind, players = turn.action_kind, set(_legal_players(turn))
    if kind == "onuw.discussion.speak_or_pass":
        return (isinstance(a.get("speak"), str) and a["speak"].strip() != "") or a.get("pass") is True
    if kind == "onuw.vote":
        return a.get("target") == -1 or a.get("target") in players
    if kind == "onuw.seer.inspect":
        if a.get("mode") == "center":
            return isinstance(a.get("indices"), list) and len(a["indices"]) >= 1
        if a.get("mode") == "player":
            return a.get("target") in players
        return False
    if kind == "onuw.robber.swap_or_decline":
        return a.get("target") in players or a.get("target") is None
    if kind == "onuw.troublemaker.swap_two_or_decline":
        if a.get("a") is None and a.get("b") is None:
            return True
        return a.get("a") in players and a.get("b") in players and a.get("a") != a.get("b")
    if kind == "onuw.doppelganger.copy_player":
        return a.get("target") in players
    if kind == "onuw.drunk.swap_center":
        return a.get("index") in (0, 1, 2)
    return True


def _gid(obj) -> str | None:
    return getattr(obj, "game_instance_id", None) if not isinstance(obj, dict) else obj.get("game_instance_id")


class ModelAgent:
    """One real LLM competitor, stateful across a game via on_event + act."""

    def __init__(self, model: str = DEFAULT_MODEL, *, max_tokens: int = 600,
                 temperature: float = 0.8, timeout: float = 60.0, notebook_limit: int = 24):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.notebook_limit = notebook_limit
        self._client: OpenAI | None = None
        self.states: dict[str, SeatState] = {}      # game_instance_id -> reconstructed view
        self.notebooks: dict[str, list[str]] = {}   # game_instance_id -> our own prior notes

    # --- SDK handlers --------------------------------------------------------
    def on_event(self, event) -> None:
        gid = _gid(event)
        if gid is None:
            return  # run-level event (e.g. run_completed) — not per-game state
        self.states.setdefault(gid, SeatState()).apply(event)

    def act(self, turn) -> dict:
        gid = turn.game_instance_id
        st = self.states.setdefault(gid, SeatState(turn.seat))
        if st.seat is None:
            st.seat = turn.seat
        notes = self.notebooks.setdefault(gid, [])
        action, reasoning = self._decide(turn, st, notes)
        notes.append(f"[{turn.phase}/{turn.action_kind}] played {json.dumps(action)} — {reasoning[:200]}")
        if len(notes) > self.notebook_limit:
            del notes[: len(notes) - self.notebook_limit]
        return {"action": action, "reasoning": reasoning}

    # --- decision ------------------------------------------------------------
    def _openai(self) -> OpenAI:
        if self._client is None:
            key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY not set in the agent's environment")
            self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
        return self._client

    def _prompt(self, turn, st: SeatState, notes: list[str]) -> str:
        instr = ACTION_INSTRUCTIONS.get(turn.action_kind, f"Take your {turn.action_kind} action.")
        parts = [st.render_context(), "", instr]
        if notes:
            parts.append("\nYOUR PRIVATE NOTEBOOK (your own earlier reasoning/moves this game; only "
                         "you see it — stay consistent):\n" + "\n".join(f"  - {n}" for n in notes))
        schema = turn.legal_action.get("schema", turn.legal_action) if turn.legal_action else {}
        parts.append("\nYour \"action\" must conform exactly to this JSON schema:\n" + json.dumps(schema))
        parts.append('\nReply with ONLY: {"reasoning": "...", "action": <the action>}')
        return "\n".join(parts)

    def _call(self, user: str) -> str:
        try:
            r = self._openai().chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                max_tokens=self.max_tokens, temperature=self.temperature, timeout=self.timeout)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            return f"<error: {type(e).__name__}>"

    def _parse(self, turn, raw: str) -> tuple[Any, str, bool]:
        obj = _extract_json(raw) or {}
        action = _coerce(turn, obj.get("action"))
        reasoning = str(obj.get("reasoning", "")).strip()
        return action, reasoning, _is_legal(turn, action)

    def _decide(self, turn, st: SeatState, notes: list[str]) -> tuple[Any, str]:
        user = self._prompt(turn, st, notes)
        action, reasoning, ok = self._parse(turn, self._call(user))
        if ok:
            return action, reasoning
        # one repair re-prompt feeding back what was wrong
        repair = user + ('\n\nYour previous reply was missing or did not match the schema. '
                         'Respond again with ONLY valid JSON and a LEGAL action.')
        action, reasoning, ok = self._parse(turn, self._call(repair))
        if ok:
            return action, reasoning
        return _fallback_action(turn), reasoning or "(fallback: model could not produce a legal action)"


# Module-level handlers for `arena-agent play ... examples/model_agent.py` (model via env).
# (Stateful CLI play also needs the runner to register on_event; the connected-sample orchestrator
# registers both directly.)
_default = ModelAgent(os.environ.get("ARENA_AGENT_MODEL", DEFAULT_MODEL))
on_event = _default.on_event
act = _default.act
