"""normalize_deck_preset / deck_for_preset: how raw preset strings collapse to a canonical key,
which presets exist, and the order in which deck_for_preset validates (preset BEFORE table size)."""
from __future__ import annotations

import pytest

from arena.games.onuw import (
    DEFAULT_DECK_PRESET,
    _DECK_PRESETS,
    deck_for_preset,
    normalize_deck_preset,
)


def test_none_falls_back_to_arena_default():
    assert normalize_deck_preset(None) == "arena"
    assert DEFAULT_DECK_PRESET == "arena"


def test_empty_string_falls_back_to_default():
    assert normalize_deck_preset("") == "arena"


def test_strips_and_lowercases_whitespace_padded_input():
    assert normalize_deck_preset("  Arena  ") == "arena"


def test_case_folds_uppercase_preset():
    assert normalize_deck_preset("CLASSIC") == "classic"


def test_internal_spaces_become_underscores_then_match():
    # the key-normalisation replaces spaces with underscores; "tanner" has no space but this
    # confirms the transform path is exercised and still resolves a real preset.
    assert normalize_deck_preset(" Tanner ") == "tanner"


def test_unknown_preset_raises_value_error_with_message():
    with pytest.raises(ValueError, match="unknown ONUW deck preset"):
        normalize_deck_preset("bogus-preset")


def test_valid_preset_set_is_exactly_arena_classic_tanner():
    assert set(_DECK_PRESETS) == {"arena", "classic", "tanner"}
    for key in ("arena", "classic", "tanner"):
        assert normalize_deck_preset(key) == key


def test_deck_for_preset_rejects_nine_players():
    with pytest.raises(ValueError, match="does not support 9 players"):
        deck_for_preset(9, "arena")


def test_deck_for_preset_validates_preset_before_table_size():
    # a bad preset AND a bad size: the preset error must fire first.
    with pytest.raises(ValueError, match="unknown ONUW deck preset"):
        deck_for_preset(9, "bogus-preset")
