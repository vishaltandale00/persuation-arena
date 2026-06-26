"""Wilson 95% CI tests: n==0 guard, point==k/n, [0,1] clamps, measured values,
the lo<=point<=hi invariant over a sweep, and z narrowing the band."""
from __future__ import annotations

import pytest

from arena.score import wilson


def test_zero_trials_returns_all_zero():
    # n==0 guard: no division, flat (lo, hi, point) == (0, 0, 0).
    assert wilson(0, 0) == (0.0, 0.0, 0.0)


def test_point_estimate_is_k_over_n():
    lo, hi, point = wilson(3, 10)
    assert point == pytest.approx(0.3)


def test_lo_clamped_at_zero_for_no_successes():
    lo, hi, point = wilson(0, 10)
    assert point == pytest.approx(0.0)
    assert lo == pytest.approx(0.0)  # max(0.0, center - half)
    assert 0.0 <= hi <= 1.0


def test_hi_clamped_at_one_for_all_successes():
    lo, hi, point = wilson(10, 10)
    assert point == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)  # min(1.0, center + half)
    assert 0.0 <= lo <= 1.0


def test_measured_value_5_of_10():
    lo, hi, point = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    assert point == pytest.approx(0.5)


def test_measured_value_3_of_4():
    lo, hi, point = wilson(3, 4)
    assert lo == pytest.approx(0.3006, abs=1e-4)
    assert hi == pytest.approx(0.9544, abs=1e-4)
    assert point == pytest.approx(0.75)


def test_measured_value_1_of_1():
    lo, hi, point = wilson(1, 1)
    assert lo == pytest.approx(0.2065, abs=1e-4)
    assert hi == pytest.approx(1.0)
    assert point == pytest.approx(1.0)


@pytest.mark.parametrize(
    "k,n",
    [(0, 1), (1, 1), (1, 2), (3, 4), (5, 10), (2, 7), (7, 7), (0, 50), (50, 50), (13, 29)],
)
def test_ordering_invariant_lo_point_hi_within_unit_interval(k, n):
    lo, hi, point = wilson(k, n)
    assert 0.0 <= lo
    assert lo <= point + 1e-12
    assert point <= hi + 1e-12
    assert hi <= 1.0


def test_smaller_z_gives_narrower_band():
    # z is the CI multiplier; a smaller z (lower confidence) tightens the interval.
    lo95, hi95, _ = wilson(5, 10, z=1.96)
    lo80, hi80, _ = wilson(5, 10, z=1.28)
    assert (hi80 - lo80) < (hi95 - lo95)
