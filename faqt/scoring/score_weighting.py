import numpy as np

SCORE_WEIGHTING_METHODS = {}


def score_weighting_func(func):
    """
    Decorator to register weighting functions
    """
    SCORE_WEIGHTING_METHODS[func.__name__] = func
    return func


@score_weighting_func
def add_weight(scores, weights, N=1.0):
    """Add weights to existing scores, and scales the weight with strength
    parameter N."""
    if not isinstance(scores, np.ndarray):
        scores = np.asarray(scores)
    if not isinstance(weights, np.ndarray):
        weights = np.asarray(weights)

    return (scores + N * weights) / (N + 1)
