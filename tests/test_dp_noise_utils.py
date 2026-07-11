import random

import pytest

from app.utils.dp_noise import (
    add_gaussian_noise,
    add_laplace_noise,
    dp_probability_ratio_bound,
    gaussian_sigma,
    make_private_count,
)


def test_gaussian_sigma_matches_formula():
    sigma = gaussian_sigma(epsilon=1.0, delta=1e-5, sensitivity=1.0)

    assert round(sigma, 6) == 4.844805


def test_dp_probability_ratio_bound():
    assert dp_probability_ratio_bound(1.0) == 2.718282


def test_gaussian_private_count_is_deterministic_with_seeded_rng():
    rng = random.Random(42)

    result = add_gaussian_noise(
        22293,
        epsilon=1.0,
        delta=1e-5,
        sensitivity=1.0,
        rng=rng,
    )

    assert result["private_count"] == 22292
    assert result["mechanism"] == "gaussian"
    assert result["sigma"] > 0
    assert result["dp_probability_ratio_bound"] == 2.718282


def test_laplace_backward_compatibility_returns_non_negative_int():
    value = add_laplace_noise(10, epsilon=1.0)

    assert isinstance(value, int)
    assert value >= 0


def test_make_private_count_rejects_invalid_mechanism():
    with pytest.raises(ValueError):
        make_private_count(100, mechanism="bad")  # type: ignore[arg-type]
