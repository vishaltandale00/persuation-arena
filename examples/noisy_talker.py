from examples.pass_agent import act as _base_act


def act(turn):
    if turn.action_kind == "onuw.discussion.speak_or_pass":
        return {
            "action": {"speak": "I want everyone to state a role claim now. Silence helps wolves."},
            "declared_reasoning": "Talkative baseline forces claims and keeps public events flowing.",
        }
    return _base_act(turn)
