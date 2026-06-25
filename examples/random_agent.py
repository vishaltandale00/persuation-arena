"""A model-free reference opponent: picks a random legal-ish action, then runs it through the SAME
canonical normalization + exact validation every harness uses, falling back to the shared
deterministic legal fallback if the pick doesn't satisfy the served schema.

This keeps the opponent correct under protocol changes (e.g. main's participant @refs and the
discussion `urgency`/`stance` fields) WITHOUT hardcoding the current shapes: the random pick is just a
starting point; `normalize_action` fills schema-required fields (like `urgency`) and `fallback_action`
guarantees a legal action when the pick is invalid. So no fake-brain opponent ever submits a stale
shape ({"speak": ...} without urgency, an integer seat, or -1) to /reply.
"""
import random

from arena.identity import NO_ONE_REF
from examples._action_schema import normalize_action, validate_action
from examples._harness_util import fallback_action

_rng = random.Random(7)


def _players(turn):
    return [p.get("ref", p.get("seat")) for p in (turn.legal_action.get("choices", {}).get("players") or [])]


def _pick(turn):
    """A random, vocabulary-correct candidate per action kind (refs from the served choices)."""
    kind = turn.action_kind
    players = _players(turn)
    if kind == "onuw.discussion.speak_or_pass":
        if _rng.random() < 0.35:
            return {"pass": True, "stance": _rng.choice(["wait", "done"])}
        # urgency is also filled by normalize_action when the schema requires it; include one anyway.
        return {"speak": "I am not certain yet, but I want pressure on quiet players.",
                "urgency": _rng.choice([1, 2, 3])}
    if kind == "onuw.vote":
        return {"target": _rng.choice(players + [NO_ONE_REF])}
    if kind == "onuw.seer.inspect":
        if players and _rng.random() < 0.5:
            return {"mode": "player", "target": _rng.choice(players)}
        return {"mode": "center", "indices": [0, 1]}
    if kind == "onuw.troublemaker.swap_two_or_decline":
        if len(players) >= 2 and _rng.random() < 0.7:
            a, b = _rng.sample(players, 2)
            return {"a": a, "b": b}
        return {"a": None, "b": None}
    if kind == "onuw.robber.swap_or_decline":
        return {"target": _rng.choice(players) if players and _rng.random() < 0.8 else None}
    if kind == "onuw.doppelganger.copy_player":
        return {"target": _rng.choice(players)} if players else {"target": None}
    if kind == "onuw.drunk.swap_center":
        return {"index": _rng.choice([0, 1, 2])}
    return {"pass": True}


def act(turn):
    # Normalize the random pick against the EXACT served schema, then validate; if it still doesn't
    # pass, use the shared deterministic legal fallback. Nothing invalid ever reaches /reply.
    action = normalize_action(turn.action_kind, turn.legal_action, _pick(turn))
    if not validate_action(turn.legal_action, action)[0]:
        action = fallback_action(turn)
    return {"action": action, "reasoning": "Reference opponent (model-free, schema-validated)."}
