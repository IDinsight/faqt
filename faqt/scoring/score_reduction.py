"""
This file contains various scoring functions that reduce the cosine similarity
between the tokens and each tag in the faq to a score per tokens/faq pair.
"""
import numpy as np

SCORE_REDUCTION_METHODS = {}


def score_reduction_func(scoring_func):
    """
    Decorator to register scoring functions
    """
    SCORE_REDUCTION_METHODS[scoring_func.__name__] = scoring_func
    return scoring_func


@score_reduction_func
def avg_min_mean(cs_values, **kwargs):
    """Original scoring: mean of avg and min"""
    return (np.mean(cs_values) + np.min(cs_values)) / 2


@score_reduction_func
def simple_mean(cs_values, **kwargs):
    """Simple mean of cs values"""
    return np.mean(cs_values)
