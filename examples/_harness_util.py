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

from arena._jsonparse import extract_json
from arena.identity import NO_ONE_REF, participant
from arena.openrouter import openrouter_client, reasoning_extra_body

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
        'accuse, defend). action = {"speak": "<what you say>", "urgency": 1|2|3} (3 = must speak now) '
        'or pass with {"pass": true, "stance": "wait"|"done"} ("done" = ready to vote).',
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

    `obj` is an SDK Event or Turn. Game scope resets between games: the game id is already globally
    unique (the arena ships `f"{run_id}_game_{NNN}"`), so it IS the key — we do NOT re-prefix run_id,
    which would double an uncapped run id into the key and risk blowing the 255-byte filename limit
    after state_path_name() even in the default mode. Run scope carries memory across games in the
    same run while keeping different runs separate. If run_id is unavailable, the game id is the
    safe fallback.
    """
    gid = getattr(obj, "game_instance_id", None)
    if reset_between_games:
        return gid  # None for run-level objects; otherwise the globally-unique game id verbatim
    run_id = getattr(obj, "run_id", None) or _run_id_from_game_id(gid)
    return run_id or gid


# Run/game ids are not length-capped upstream (--run-id / POST /api/runs), so a raw key can exceed
# common 255-byte filename limits. Bound the readable stem and lean on the hash suffix for identity.
_STATE_STEM_MAX = 96


def state_path_name(key: str) -> str:
    """Make a state key safe as one local path segment, INJECTIVELY and within filename limits.

    Sanitizing alone is many-to-one ('run:a' and 'run_a' both collapse to 'run_a'), which would
    point two distinct state keys at the same memory file/workspace and corrupt run separation. So
    we suffix a short stable hash of the EXACT key: the (length-bounded) sanitized stem stays for
    humans, while the hash guarantees distinct keys never share a path even when the bounded stems
    collide. The bound keeps the segment well under the 255-byte filesystem limit for any key.
    """
    sanitized = (re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_") or "state")[:_STATE_STEM_MAX]
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
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set in the harness's environment")
    try:
        # Shared OpenRouter client; compression is left OFF for harnesses (the arena's own
        # client opts in). Keeping it explicit means the unified helper never silently changes
        # the harness request shape.
        r = openrouter_client(key).chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            extra_body=reasoning_extra_body(DEFAULT_REASONING_EFFORT, compression=False),
        )
        message = r.choices[0].message
        raw = (_message_field(message, "content") or "").strip()
        return _assistant_message(raw, message)
    except Exception as e:
        return _assistant_message(f"<error: {type(e).__name__}>")


# Engine-as-source-of-truth: every harness affordance below is derived from the turn's shipped
# `legal_action` (the `choices`/`schema` the engine emitted), NOT from hand-maintained per-role
# constants. The engine is free to tighten an action's shape (e.g. discussion now REQUIRES an
# urgency/stance); these helpers read that shape off the turn so the harness can't drift from it.
# jsonschema is not installed, so the few stdlib checks here interpret the shipped `choices`.


def _choices(turn) -> dict:
    la = turn.legal_action or {}
    ch = la.get("choices")
    return ch if isinstance(ch, dict) else {}


def _schema(turn) -> dict:
    la = turn.legal_action or {}
    sch = la.get("schema")
    return sch if isinstance(sch, dict) else {}


def _matches_schema(schema: dict, value: Any) -> bool:
    """A tiny stdlib validator for the JSON-Schema SUBSET the ONUW engine actually ships
    (object/oneOf/required/properties/additionalProperties/enum/type + array
    items/minItems/maxItems/uniqueItems). jsonschema is not installed; we only need to cover what
    the engine emits, and an empty {} schema accepts anything (matching jsonschema)."""
    if not schema:
        return True
    if "oneOf" in schema:
        return sum(1 for s in schema["oneOf"] if _matches_schema(s, value)) == 1
    if "enum" in schema:
        if value not in schema["enum"]:
            return False
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return False
        for req in schema.get("required", []):
            if req not in value:
                return False
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            if any(k not in props for k in value):
                return False
        return all(_matches_schema(props[k], value[k]) for k in value if k in props)
    if t == "array":
        if not isinstance(value, list):
            return False
        if "minItems" in schema and len(value) < schema["minItems"]:
            return False
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return False
        if schema.get("uniqueItems") and len(value) != len(set(map(_hashable, value))):
            return False
        item_schema = schema.get("items")
        return item_schema is None or all(_matches_schema(item_schema, v) for v in value)
    if t == "string":
        if not isinstance(value, str):
            return False
        if "minLength" in schema and len(value) < schema["minLength"]:
            return False
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return False
        return True
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "null":
        return value is None
    return True


def _hashable(v: Any) -> Any:
    return tuple(v) if isinstance(v, list) else v


def _legal_players(turn) -> list:
    return [p.get("ref", p.get("seat")) for p in (_choices(turn).get("players") or [])]


def _coerce(turn, a: Any) -> Any:
    # A bare string in discussion is a shorthand for speaking; the engine now requires an urgency
    # on every speak, so coerce to the lowest urgency rather than emitting a bare {speak:...}.
    if turn.action_kind == "onuw.discussion.speak_or_pass" and isinstance(a, str):
        s = a.strip()
        if s.lower() in ("pass", "(pass)", ""):
            return {"pass": True, "stance": "done"}
        return {"speak": a, "urgency": 1}
    return a


def _is_legal(turn, a: Any) -> bool:
    if not isinstance(a, dict):
        return False
    # The engine ships the authoritative JSON-Schema for the turn; when present it is the source
    # of truth (it already enumerates legal player refs, center indices, urgency 1..3, stances,
    # and requires the fields the engine now mandates — e.g. urgency on every speak).
    schema = _schema(turn)
    if schema:
        return _matches_schema(schema, a)
    # No schema shipped (older/fake turns): fall back to choices-driven shape checks.
    kind = turn.action_kind
    choices = _choices(turn)
    players = set(_legal_players(turn))
    if kind == "onuw.discussion.speak_or_pass":
        if a.get("pass") is True:
            stances = set(choices.get("stances") or [])
            return a.get("stance") in stances if stances else True
        return isinstance(a.get("speak"), str) and a["speak"].strip() != ""
    if kind == "onuw.vote":
        no_one = choices.get("no_one", NO_ONE_REF)
        return a.get("target") in {no_one, NO_ONE_REF, -1} or a.get("target") in players
    if kind == "onuw.seer.inspect":
        center = set(choices.get("center") or [])
        if a.get("mode") == "center":
            idx = a.get("indices")
            return (isinstance(idx, list) and len(idx) == 2 and idx[0] != idx[1]
                    and all(i in center for i in idx))
        if a.get("mode") == "player":
            return a.get("target") in players
        return False
    if kind == "onuw.robber.swap_or_decline":
        if a.get("target") is None:
            return bool(choices.get("decline"))
        return a.get("target") in players
    if kind == "onuw.troublemaker.swap_two_or_decline":
        if a.get("a") is None and a.get("b") is None:
            return bool(choices.get("decline"))
        return a.get("a") in players and a.get("b") in players and a.get("a") != a.get("b")
    if kind == "onuw.doppelganger.copy_player":
        return a.get("target") in players
    if kind == "onuw.drunk.swap_center":
        return a.get("index") in set(choices.get("center") or [])
    return True


def _fallback_action(turn) -> dict:
    """A guaranteed-legal action, built from the engine's shipped `choices` so it always matches
    the current action shape (mirrors the engine's own `default_wire_action`, which is not
    forwarded to the harness on the turn)."""
    kind = turn.action_kind
    choices = _choices(turn)
    players = _legal_players(turn)
    center = list(choices.get("center") or [])
    if kind == "onuw.vote":
        return {"target": choices.get("no_one", NO_ONE_REF)}
    if kind == "onuw.seer.inspect":
        return {"mode": "center", "indices": center[:2] if len(center) >= 2 else [0, 1]}
    if kind == "onuw.troublemaker.swap_two_or_decline":
        return {"a": None, "b": None}
    if kind == "onuw.robber.swap_or_decline":
        return {"target": None} if choices.get("decline") else {"target": players[0] if players else None}
    if kind == "onuw.doppelganger.copy_player":
        return {"target": players[0] if players else 0}
    if kind == "onuw.drunk.swap_center":
        return {"index": center[0] if center else 0}
    stances = choices.get("stances") or ["done"]
    return {"pass": True, "stance": "done" if "done" in stances else stances[0]}


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
    obj = extract_json(raw_text) or {}
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
