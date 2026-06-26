from arena.wolf_profiles import BASE_SYSTEM, PROFILE_INSTRUCTIONS, prompt_for


def test_named_profiles_are_distinct() -> None:
    prompts = {name: prompt_for(name) for name in ("base", "bayes", "charisma", "chaos")}
    assert len(set(prompts.values())) == 4
    assert all('"declared_reasoning"' in prompt and '"action"' in prompt for prompt in prompts.values())


def test_base_profile_preserves_original_prompt() -> None:
    assert prompt_for("base") == BASE_SYSTEM


def test_unknown_profile_falls_back_to_base_policy() -> None:
    assert prompt_for("does-not-exist") == prompt_for("base")


def test_profile_lookup_is_case_insensitive() -> None:
    assert PROFILE_INSTRUCTIONS["bayes"] in prompt_for("BaYeS")
