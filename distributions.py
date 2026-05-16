"""
Support functions for Factory Floor Data Simulator
===================================================
Provides stochastic sampling utilities for unit production volumes
and defect rates used in the simulation.
"""


import numpy as np

# Module-level RNG instance.
_RNG = np.random.default_rng()

# Safety cap for rejection sampling — prevents infinite loops when
# bounds are too tight relative to the distribution.
_MAX_REJECTION_ITER = 10_000

def sample_production_units(size: int = 1_000, low: int = 800, high: int = 1_200, sigma: float = 0.1) -> np.ndarray:
    """
    Generate an array of integer production unit counts drawn from a
    log-normal distribution, bounded between ``low`` and ``high``.

    The log-normal mean (mu) is derived automatically from the midpoint
    of [low, high], so the distribution stays centered on the range
    regardless of what bounds the caller passes.

    Parameters
    ----------
    size : int
        Number of samples to return.
    low : int
        Minimum acceptable value (inclusive).
    high : int
        Maximum acceptable value (inclusive).
    sigma : float
        Shape parameter of the log-normal distribution.
        Smaller values produce a tighter spread around the midpoint.

    Returns
    -------
    numpy.ndarray of dtype int

    Raises
    ------
    RuntimeError
        If the sampler cannot collect enough valid samples within
        _MAX_REJECTION_ITER iterations, which indicates the bounds
        are too tight for the given sigma.
    """

    # Derive mu from the midpoint of the requested range
    mu = np.log((low + high) / 2)

    samples = []
    iterations = 0

    while len(samples) < size:
        iterations += 1

        if iterations > _MAX_REJECTION_ITER:
            raise RuntimeError(
                f"Could not collect {size} samples within {_MAX_REJECTION_ITER} "
                f"iterations. Consider widening [low={low}, high={high}] or "
                f"increasing sigma={sigma}."
            )

        # Draw a batch at once instead of one value per iteration
        batch = _RNG.lognormal(mu, sigma, size=size)
        valid = batch[(batch >= low) & (batch <= high)]
        samples.extend(valid.tolist())

    # Trim to exactly the requested size (extend may have overshot)
    return np.array(samples[:size], dtype=int)

def sample_defect_rate(mean: float = 0.025, std: float = 0.005) -> float:
    """
    Return a single random defect rate drawn from a truncated normal
    distribution, guaranteed to lie within [0, 1].

    A beta distribution would be the theoretically correct choice for
    a rate in [0, 1], but a clipped normal is a reasonable approximation
    when mean is well away from 0 and 1 (as is typical for defect rates).

    Parameters
    ----------
    mean : float
        Expected defect rate (e.g. 0.025 = 2,5 %).
    std : float
        Standard deviation around the mean.

    Returns
    -------
    float
        A defect rate in [0, 1].
    """

    rate = _RNG.normal(mean, std)

    # Clip is intentional and documented: the normal distribution has
    # unbounded tails, so we explicitly constrain to a valid rate range.
    return float(np.clip(rate, 0.0, 1.0))