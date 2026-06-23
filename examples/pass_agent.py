def _first_player(turn):
    players = turn.legal_action.get("choices", {}).get("players") or []
    return players[0]["seat"] if players else 0


def act(turn):
    kind = turn.action_kind
    if kind == "onuw.discussion.speak_or_pass":
        return {"action": {"pass": True}, "reasoning": "Pass-only baseline."}
    if kind == "onuw.vote":
        return {"action": {"target": -1}, "reasoning": "Pass-only baseline votes for no one."}
    if kind == "onuw.seer.inspect":
        return {"action": {"mode": "center", "indices": [0, 1]}, "reasoning": "Default center look."}
    if kind == "onuw.troublemaker.swap_two_or_decline":
        return {"action": {"a": None, "b": None}, "reasoning": "Decline action."}
    if kind in {"onuw.doppelganger.copy_player", "onuw.robber.swap_or_decline"}:
        return {"action": {"target": _first_player(turn)}, "reasoning": "Use first legal player."}
    if kind == "onuw.drunk.swap_center":
        return {"action": {"index": 0}, "reasoning": "Use first center card."}
    return {"action": {"pass": True}, "reasoning": "Fallback pass."}
