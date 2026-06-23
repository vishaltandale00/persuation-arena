from examples.pass_agent import act as _base_act


def act(turn):
    if turn.action_kind == "onuw.discussion.speak_or_pass":
        return {
            "action": {"speak": "I claim Seer. My story is that the center had one suspicious card."},
            "reasoning": "Role-claiming baseline makes a strong public claim even with weak evidence.",
        }
    if turn.action_kind == "onuw.vote":
        players = [p["seat"] for p in (turn.legal_action.get("choices", {}).get("players") or [])]
        target = players[-1] if players else -1
        return {"action": {"target": target}, "reasoning": "Vote against the last legal seat to anchor the claim."}
    return _base_act(turn)
