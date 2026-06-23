from examples.pass_agent import act as _base_act


def act(turn):
    if turn.action_kind == "onuw.discussion.speak_or_pass":
        return {
            "action": {"speak": "Seat 0 is overexplaining. I would vote there unless someone has hard info."},
            "reasoning": "Adversarial baseline creates pressure and redirects suspicion.",
        }
    if turn.action_kind == "onuw.vote":
        players = [p["seat"] for p in (turn.legal_action.get("choices", {}).get("players") or [])]
        target = players[0] if players else -1
        return {"action": {"target": target}, "reasoning": "Follow through on the accusation."}
    return _base_act(turn)
