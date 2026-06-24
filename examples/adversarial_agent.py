from examples.pass_agent import act as _base_act
from arena.identity import NO_ONE_REF


def act(turn):
    if turn.action_kind == "onuw.discussion.speak_or_pass":
        return {
            "action": {"speak": "The loudest overexplainer is suspicious. I would vote there unless someone has hard info."},
            "reasoning": "Adversarial baseline creates pressure and redirects suspicion.",
        }
    if turn.action_kind == "onuw.vote":
        players = [p.get("ref", p.get("seat")) for p in (turn.legal_action.get("choices", {}).get("players") or [])]
        target = players[0] if players else NO_ONE_REF
        return {"action": {"target": target}, "reasoning": "Follow through on the accusation."}
    return _base_act(turn)
