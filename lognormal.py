import numpy as np


def sample_(size: int = 10, low: int = 800, high: int = 1200, mu=np.log(1000), sigma: float=0.1):
    samples = []
    while len(samples) < size:
        x = np.random.lognormal(mu, sigma, size=1)
        if low <= x <= high:
            samples.append(x.item())

    return np.array(samples, dtype=int)

def sample_error_rate_(mean=0.03, std=0.005):
    p = np.random.default_rng().normal(mean, std, size=1)
    return np.clip(p, a_min=0, a_max=1).item()





