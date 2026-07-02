import numpy as np


def add_laplace_noise(count: int, epsilon: float = 1.0) -> int:
    if count <= 0:
        return 0

    noise = np.random.laplace(loc=0.0, scale=1.0 / epsilon)
    noisy_count = int(round(count + noise))

    return max(noisy_count, 0)
