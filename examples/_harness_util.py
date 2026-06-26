"""Shared plumbing for example harnesses, so each harness file is JUST its memory strategy.

Building memory from the delta event stream is the harness's job: the SDK delivers each event once
and the server never re-serves old ones, so a harness that keeps nothing plays blind. These helpers
are the boring parts every harness shares — rendering an event to text, and turning the messages a
harness assembled (from its own memory) into a validated action.

decide(model, messages, turn) -> (action, reasoning, assistant_message):
  call the LLM with the messages you built from YOUR memory, parse {declared_reasoning, action}, validate
  against the turn's legal_action, repair once, else fall back to a legal action so a seat never
  forfeits on bad output. (The server validates authoritatively too.)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from openai import OpenAI

from arena.identity import NO_ONE_REF, participant

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
RESET_BETWEEN_GAMES_ENV = "ARENA_AGENT_RESET_BETWEEN_GAMES"
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
DEFAULT_MAX_TOKENS = int(os.environ.get("ARENA_AGENT_MAX_TOKENS", os.environ.get("ARENA_MAX_TOKENS_PER_TURN", "4000")))
DEFAULT_TEMPERATURE = float(os.environ.get("ARENA_AGENT_TEMPERATURE", os.environ.get("ARENA_TEMPERATURE", "0.8")))
DEFAULT_TIMEOUT = float(os.environ.get("ARENA_AGENT_REQUEST_TIMEOUT_S", os.environ.get("ARENA_REQUEST_TIMEOUT_S", "60")))
DEFAULT_REASONING_EFFORT = os.environ.get("ARENA_AGENT_REASONING_EFFORT", os.environ.get("ARENA_REASONING_EFFORT", "medium")).strip().lower()
if DEFAULT_REASONING_EFFORT not in REASONING_EFFORTS:
    DEFAULT_REASONING_EFFORT = "medium"

SYSTEM = (
    "You are a sharp, competitive player of a hidden-role social-deduction game (One Night Ultimate "
    "Werewolf). You receive the game as a stream of events; track what you learn, reason about who "
    "is lying, and act for your team. When asked to act, reply with ONE JSON object and nothing "
    'else: {"declared_reasoning": "<your private thinking>", "action": <the action>}.'
)

ACTION_INSTRUCTIONS = {
    "onuw.discussion.speak_or_pass":
        'Speak to the whole table — be persuasive for YOUR team (claim a role, share or fake info, '
        'accuse, defend). action = {"speak": "<what you say>"} or {"pass": true}.',
    "onuw.vote":
        f'Vote for who should be eliminated. action = {{"target": "@participant"}} or {{"target": "{NO_ONE_REF}"}} for no one.',
    "onuw.seer.inspect":
        'NIGHT (Seer): action = {"mode":"player","target":"@participant"} to see a player\'s card, OR '
        '{"mode":"center","indices":[a,b]} to see two center cards.',
    "onuw.robber.swap_or_decline":
        'NIGHT (Robber): action = {"target":"@participant"} to swap and learn your new role, or {"target": null}.',
    "onuw.troublemaker.swap_two_or_decline":
        'NIGHT (Troublemaker): action = {"a":"@participant","b":"@participant"} to swap two others (unseen), or '
        '{"a": null, "b": null}.',
    "onuw.drunk.swap_center":
        'NIGHT (Drunk): action = {"index": 0|1|2} to blindly swap with that center card.',
    "onuw.doppelganger.copy_player":
        'NIGHT (Doppelganger): action = {"target":"@participant"} to copy that player\'s role.',
}

_client: OpenAI | None = None


def reset_between_games_from_env(default: bool = True) -> bool:
    """Return the memory-scope toggle shared by the reference stateful harnesses.

    Default true means one memory/session per game. Set ARENA_AGENT_RESET_BETWEEN_GAMES=0 to keep
    one memory/session for the whole run. A new run still gets a separate state key when run_id is
    available (directly on the SDK turn/event, or recovered from the game id prefix).
    """
    raw = os.environ.get(RESET_BETWEEN_GAMES_ENV)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"0", "false", "no", "off", "run", "per-run", "per_run"}:
        return False
    if value in {"1", "true", "yes", "on", "game", "per-game", "per_game"}:
        return True
    return default


def _run_id_from_game_id(game_instance_id: str | None) -> str | None:
    """Recover the run id from a game id of the form '<run_id>_game_<NNN>' (the arena's shape)."""
    if not game_instance_id:
        return None
    if "_game_" in game_instance_id:
        return game_instance_id.rsplit("_game_", 1)[0]
    return None


def state_key(obj, reset_between_games: bool = True) -> str | None:
    """Key harness-owned memory by game or run.

    `obj` is an SDK Event or Turn. Game scope resets between games (one key per game, still
    namespaced by run when known). Run scope carries memory across games in the same run while
    keeping different runs separate. If run_id is unavailable, the game id is the safe fallback.
    """
    gid = getattr(obj, "game_instance_id", None)
    run_id = getattr(obj, "run_id", None) or _run_id_from_game_id(gid)
    if reset_between_games:
        if gid is None:
            return None
        return f"{run_id}:{gid}" if run_id else gid
    return run_id or gid


def state_path_name(key: str) -> str:
    """Make a state key safe as one local path segment, INJECTIVELY.

    Sanitizing alone is many-to-one ('run:a' and 'run_a' both collapse to 'run_a'), which would
    point two distinct state keys at the same memory file/workspace and corrupt run separation. So
    we suffix a short stable hash of the exact key: the readable sanitized stem stays for humans,
    while the hash guarantees distinct keys never share a path.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_") or "state"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{sanitized}-{digest}"


def _message_field(message: Any, field: str) -> Any:
    if isinstance(message, dict):
        return message.get(field)
    return getattr(message, field, None)


def _assistant_message(raw: str, message: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": raw}
    reasoning_details = _message_field(message, "reasoning_details") if message is not None else None
    reasoning = _message_field(message, "reasoning") if message is not None else None
    if reasoning_details is not None:
        out["reasoning_details"] = reasoning_details
    elif reasoning is not None:
        out["reasoning"] = reasoning
    return out


def _llm(
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    global _client
    if _client is None:
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set in the harness's environment")
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
    try:
        r = _client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            extra_body={
                "reasoning": {"effort": DEFAULT_REASONING_EFFORT, "exclude": False},
            },
        )
        message = r.choices[0].message
        raw = (_message_field(message, "content") or "").strip()
        return _assistant_message(raw, message)
    except Exception as e:
        return _assistant_message(f"<error: {type(e).__name__}>")


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


def _legal_players(turn) -> list:
    return [p.get("ref", p.get("seat")) for p in (turn.legal_action.get("choices", {}).get("players") or [])]


def _coerce(turn, a: Any) -> Any:
    if turn.action_kind == "onuw.discussion.speak_or_pass" and isinstance(a, str):
        return {"pass": True} if a.strip().lower() in ("pass", "(pass)", "") else {"speak": a}
    return a


def _is_legal(turn, a: Any) -> bool:
    if not isinstance(a, dict):
        return False
    kind, players = turn.action_kind, set(_legal_players(turn))
    if kind == "onuw.discussion.speak_or_pass":
        return (isinstance(a.get("speak"), str) and a["speak"].strip() != "") or a.get("pass") is True
    if kind == "onuw.vote":
        return a.get("target") in {NO_ONE_REF, -1} or a.get("target") in players
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


def _fallback_action(turn) -> dict:
    kind, players = turn.action_kind, _legal_players(turn)
    if kind == "onuw.vote":
        return {"target": NO_ONE_REF}
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


def render_event(event) -> str:
    """One readable line per event — what a harness appends to its memory."""
    et = getattr(event, "type", None)
    p = getattr(event, "payload", None) or {}
    def who(value) -> str:
        if isinstance(value, dict):
            return f"{value.get('name')} ({value.get('ref')})"
        return f"Participant {int(value) + 1}" if isinstance(value, int) else "Participant"
    if et == "game_setup":
        participants = p.get("participants")
        if participants is None:
            participants = [participant(v) for v in (p.get("roster") or {}).values()]
        roster = ", ".join(who(x) for x in participants)
        return (f"SETUP: {p.get('n')}-player ONUW. Players: {roster}. Cards in play (public): "
                f"{p.get('deck')}. {p.get('center_count')} face-down center cards. "
                f"Win condition: {p.get('win_condition')}")
    if et == "role_info":
        return f"YOUR ROLE: you are {who(p.get('participant') or p.get('seat'))}; your dealt card is {p.get('role')}."
    if et == "night_observation":
        return f"NIGHT (private to you): {p.get('text')}"
    if et == "speech":
        return f"{who(p.get('actor') or p.get('actor_seat'))} says: {p.get('text')}"
    if et == "pass":
        return f"{who(p.get('actor') or p.get('actor_seat'))} stays silent."
    if et in ("phase_started", "phase_ended"):
        return f"PHASE {et.split('_')[1]}: {p.get('phase')} — {p.get('text', '')}"
    if et == "vote_revealed":
        return f"{who(p.get('actor') or p.get('actor_seat'))} voted for {who(p.get('target'))}."
    if et == "game_result":
        return f"RESULT: {p.get('text')}"
    return f"{et}: {json.dumps(p)}"


def action_request(turn) -> str:
    """The user message a harness appends when it's the agent's turn."""
    instr = ACTION_INSTRUCTIONS.get(turn.action_kind, f"Take your {turn.action_kind} action.")
    schema = turn.legal_action.get("schema", turn.legal_action) if turn.legal_action else {}
    return (f"YOUR TURN ({turn.action_kind}). {instr}\n"
            f"Your \"action\" must match this JSON schema exactly:\n{json.dumps(schema)}\n"
            'Reply with ONLY {"declared_reasoning": "...", "action": <the action>}.')


REPAIR_MESSAGE = ("That action was missing or illegal. "
                  "Reply again with ONLY valid JSON and a LEGAL action.")


def interpret(turn, raw_text: str) -> tuple[Any, str, bool]:
    """Parse one brain's raw text into (action, reasoning, is_legal). Pure: no model call, no
    mutation. Every harness brain (OpenRouter chat, codex, opencode, Claude Agent SDK) funnels its
    output through here, so JSON extraction, coercion, and legality live in exactly one place."""
    obj = _extract_json(raw_text) or {}
    action = _coerce(turn, obj.get("action"))
    reasoning = str(obj.get("declared_reasoning", obj.get("reasoning", ""))).strip()
    return action, reasoning, _is_legal(turn, action)


def fallback_action(turn) -> dict:
    """A guaranteed-legal action so a seat never forfeits when a brain yields nothing usable."""
    return _fallback_action(turn)


def decide(model: str, messages: list[dict], turn) -> tuple[Any, str, dict[str, Any]]:
    """LLM call + validate + one repair + legal fallback. `messages` is whatever the harness built
    from its own memory; this never mutates it."""
    assistant = _llm(model, messages)
    raw = str(assistant.get("content") or "")
    action, reasoning, legal = interpret(turn, raw)
    if legal:
        return action, reasoning, assistant
    repair = list(messages) + [assistant, {"role": "user", "content": REPAIR_MESSAGE}]
    assistant2 = _llm(model, repair)
    raw2 = str(assistant2.get("content") or "")
    action2, reasoning2, legal2 = interpret(turn, raw2)
    if legal2:
        return action2, reasoning2, assistant2
    return fallback_action(turn), reasoning or "(fallback: no legal action produced)", assistant
