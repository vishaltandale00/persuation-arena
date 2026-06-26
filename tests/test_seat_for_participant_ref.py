from __future__ import annotations

import pytest

from arena.identity import NO_ONE_REF, seat_for_participant_ref

NAMES = {0: "Alice", 1: "Bob", 2: "Cora"}
CANDIDATES = [0, 1, 2]


def test_ref_resolves_to_seat():
    assert seat_for_participant_ref("@bob", NAMES, CANDIDATES) == 1


def test_full_name_resolves_to_seat():
    assert seat_for_participant_ref("Cora", NAMES, CANDIDATES) == 2


def test_bare_handle_resolves_to_seat():
    assert seat_for_participant_ref("alice", NAMES, CANDIDATES) == 0


def test_int_passthrough_when_in_candidates():
    assert seat_for_participant_ref(1, NAMES, CANDIDATES) == 1


def test_int_not_in_candidates_raises_bad_target():
    with pytest.raises(ValueError, match="bad target"):
        seat_for_participant_ref(9, NAMES, CANDIDATES)


def test_bool_true_is_rejected_not_coerced_to_seat_one():
    with pytest.raises(ValueError, match="bad target"):
        seat_for_participant_ref(True, NAMES, CANDIDATES)


def test_bool_false_is_rejected_not_coerced_to_seat_zero():
    with pytest.raises(ValueError, match="bad target"):
        seat_for_participant_ref(False, NAMES, CANDIDATES)


def test_unknown_ref_raises_bad_target():
    with pytest.raises(ValueError, match="bad target"):
        seat_for_participant_ref("@nobody", NAMES, CANDIDATES)


@pytest.mark.parametrize("value", [-1, NO_ONE_REF, "none", "no one"])
def test_allow_no_one_maps_to_minus_one(value):
    assert seat_for_participant_ref(value, NAMES, CANDIDATES, allow_no_one=True) == -1


def test_no_one_ref_rejected_when_allow_no_one_false():
    with pytest.raises(ValueError, match="bad target"):
        seat_for_participant_ref(NO_ONE_REF, NAMES, CANDIDATES)
