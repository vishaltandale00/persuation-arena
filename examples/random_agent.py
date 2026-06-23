import random


_rng = random.Random(7)


def _players(turn):
    return [p["seat"] for p in (turn.legal_action.get("choices", {}).get("players") or [])]


def act(turn):
    kind = turn.action_kind
    players = _players(turn)
    if kind == "onuw.discussion.speak_or_pass":
        if _rng.random() < 0.35:
            return {"action": {"pass": True}, "reasoning": "Randomly chose silence."}
        return {"action": {"speak": "I am not certain yet, but I want pressure on quiet seats."},
                "reasoning": "Random baseline chose to speak."}
    if kind == "onuw.vote":
        return {"action": {"target": _rng.choice(players + [-1])}, "reasoning": "Random legal vote."}
    if kind == "onuw.seer.inspect":
        if players and _rng.random() < 0.5:
            return {"action": {"mode": "player", "target": _rng.choice(players)}, "reasoning": "Random player inspect."}
        return {"action": {"mode": "center", "indices": [0, 1]}, "reasoning": "Random center inspect."}
    if kind == "onuw.troublemaker.swap_two_or_decline":
        if len(players) >= 2 and _rng.random() < 0.7:
            a, b = _rng.sample(players, 2)
            return {"action": {"a": a, "b": b}, "reasoning": "Random swap."}
        return {"action": {"a": None, "b": None}, "reasoning": "Random decline."}
    if kind == "onuw.robber.swap_or_decline":
        return {"action": {"target": _rng.choice(players) if players and _rng.random() < 0.8 else None},
                "reasoning": "Random robber choice."}
    if kind == "onuw.doppelganger.copy_player":
        return {"action": {"target": _rng.choice(players)}, "reasoning": "Random copy target."}
    if kind == "onuw.drunk.swap_center":
        return {"action": {"index": _rng.choice([0, 1, 2])}, "reasoning": "Random center card."}
    return {"action": {"pass": True}, "reasoning": "Fallback pass."}
