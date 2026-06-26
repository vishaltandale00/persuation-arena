from __future__ import annotations

from arena.identity import participant, public_ref


def test_public_ref_internal_space_becomes_single_hyphen():
    assert public_ref("Alice Smith") == "@alice-smith"


def test_public_ref_whitespace_only_is_empty_string_not_at():
    assert public_ref("   ") == ""


def test_public_ref_punctuation_runs_collapse_to_single_hyphen():
    assert public_ref("Bob!!!  Jones") == "@bob-jones"


def test_public_ref_pure_digits_are_valid_handle():
    assert public_ref("123") == "@123"


def test_participant_collapses_internal_whitespace():
    assert participant("  Bob   Jones ") == {"name": "Bob Jones", "ref": "@bob-jones"}
