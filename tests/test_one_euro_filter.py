import math

import numpy as np
import pytest

from ahma.core.metrics import OneEuroFilter


def test_smoothing_factor_matches_formula():
    filt = OneEuroFilter(t0=0.0, x0=0.0)
    t_e, cutoff = 0.5, 2.0
    r = 2 * math.pi * cutoff * t_e

    assert filt.smoothing_factor(t_e, cutoff) == pytest.approx(r / (r + 1))


def test_non_positive_delta_returns_previous_value():
    filt = OneEuroFilter(t0=1.0, x0=5.0)

    assert filt(1.0, 10.0) == 5.0
    assert filt(0.5, 10.0) == 5.0


def test_constant_signal_stays_constant():
    filt = OneEuroFilter(t0=0.0, x0=1.0)

    for i in range(1, 60):
        out = filt(i / 30.0, 1.0)

    assert out == pytest.approx(1.0, abs=1e-6)


def test_filter_reduces_jitter():
    rng = np.random.default_rng(42)
    signal = 10.0 + rng.normal(0.0, 1.0, size=200)
    filt = OneEuroFilter(t0=0.0, x0=float(signal[0]), min_cutoff=1.0, beta=0.007)

    smoothed = [filt(i / 30.0, float(value)) for i, value in enumerate(signal[1:], start=1)]

    assert np.std(smoothed[10:]) < np.std(signal[10:]) * 0.5


def test_filter_tracks_step_change():
    filt = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=0.007)

    for i in range(1, 90):
        out = filt(i / 30.0, 10.0)

    assert out == pytest.approx(10.0, abs=0.5)
