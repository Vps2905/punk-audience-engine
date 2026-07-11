from __future__ import annotations

import math
import random
from typing import Any, Dict, Literal, Optional


DPMechanism = Literal["laplace", "gaussian"]


def validate_dp_params(
    *,
    epsilon: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
) -> None:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be between 0 and 1")

    if sensitivity <= 0:
        raise ValueError("sensitivity must be > 0")


def gaussian_sigma(
    *,
    epsilon: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
) -> float:
    """
    Gaussian mechanism for approximate (epsilon, delta)-DP.

    sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    """
    validate_dp_params(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def dp_probability_ratio_bound(epsilon: float) -> float:
    """
    DP probability ratio bound: e^epsilon.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    return round(math.exp(epsilon), 6)


def laplace_noise(
    *,
    epsilon: float,
    sensitivity: float = 1.0,
    rng: Optional[random.Random] = None,
) -> float:
    """
    Pure epsilon-DP Laplace noise.

    noise ~ Laplace(0, sensitivity / epsilon)
    """
    validate_dp_params(epsilon=epsilon, sensitivity=sensitivity)

    rng = rng or random.Random()
    scale = sensitivity / epsilon

    u = rng.random() - 0.5

    if u == 0:
        return 0.0

    sign = 1.0 if u > 0 else -1.0
    return -scale * sign * math.log(1 - 2 * abs(u))


def gaussian_noise(
    *,
    epsilon: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    rng: Optional[random.Random] = None,
) -> float:
    """
    Approximate (epsilon, delta)-DP Gaussian noise.

    noise ~ Normal(0, sigma^2)
    """
    rng = rng or random.Random()
    sigma = gaussian_sigma(
        epsilon=epsilon,
        delta=delta,
        sensitivity=sensitivity,
    )

    return rng.gauss(0.0, sigma)


def make_private_count(
    count: int | float,
    *,
    mechanism: DPMechanism = "gaussian",
    epsilon: float = 1.0,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Convert an internal bounded count into an external-safe private_count.

    Raw count should remain internal. External output should expose only
    private_count and DP metadata.
    """
    validate_dp_params(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

    internal_count = max(0.0, float(count))
    mechanism = mechanism.lower()  # type: ignore[assignment]

    if mechanism == "laplace":
        noise = laplace_noise(
            epsilon=epsilon,
            sensitivity=sensitivity,
            rng=rng,
        )
        sigma = None
        formula = "private_count = real_count + Laplace(0, sensitivity / epsilon)"

    elif mechanism == "gaussian":
        sigma = gaussian_sigma(
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
        )
        noise = gaussian_noise(
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            rng=rng,
        )
        formula = "private_count = real_count + Normal(0, sigma^2)"

    else:
        raise ValueError("mechanism must be 'laplace' or 'gaussian'")

    private_count = max(0, int(round(internal_count + noise)))

    return {
        "private_count": private_count,
        "mechanism": mechanism,
        "epsilon": epsilon,
        "delta": delta if mechanism == "gaussian" else None,
        "sensitivity": sensitivity,
        "sigma": sigma,
        "noise_added": noise,
        "formula": formula,
        "dp_probability_ratio_bound": dp_probability_ratio_bound(epsilon),
        "privacy_note": "Expose private_count externally. Keep raw count/noise internal unless debugging.",
    }


def add_laplace_noise(count: int, epsilon: float = 1.0) -> int:
    """
    Backward-compatible helper for older services.
    """
    result = make_private_count(
        count,
        mechanism="laplace",
        epsilon=epsilon,
        delta=1e-5,
        sensitivity=1.0,
    )

    return int(result["private_count"])


def add_gaussian_noise(
    count: int,
    *,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Gaussian DP helper for privacy-safe count release.
    """
    return make_private_count(
        count,
        mechanism="gaussian",
        epsilon=epsilon,
        delta=delta,
        sensitivity=sensitivity,
        rng=rng,
    )
