"""
This file contains various scoring functions that reduce the cosine similarity
between the message and each tag in the faq to a score per message/faq pair.
"""
import numpy as np

SCORE_REDUCTION_METHODS = {}

# TODO: refactor the module to contain different options for functions that map
#  `(List[vector], List[vector])` to `float`
#  For example, reduction_function([cs_nearest_k_percent_average(...)]),
#  or word_movers_distance(...)


def get_faq_scores_for_message(
    tag_scores,
    weights,
    reduction_func,
    **reduction_func_arg,
):
    """
    Returns tag_scores for the inbound vectors against each faq

    Parameters
    ----------
    tag_scores: List[Dict[str, float]]
        A list with an item for each FAQ. Each item is a dictionary with keys
        as tags
        and values as the cosine similarity metric between incoming message
        and the tag
    weights: List[float]
        weights for each faq content
    reduction_func: str
        Name of one of the score reduction functions that reduce cosine
        similarity
        between message and each faq tag to a score for each message/faq
        combination.
    reduction_func_arg: Dict
        Additional arguments to be send to `reduction_func`

    Returns
    -------
    Dict[int, Dict]
        A Dictionary with `faq_id` as key. Values: faq details including tag_scores
        for each tag and an `overall_score`
    """
    scoring = []

    scoring_method = SCORE_REDUCTION_METHODS.get(reduction_func)
    if scoring_method is None:
        all_scoring_methods = SCORE_REDUCTION_METHODS.keys()
        raise NotImplementedError(
            (
                f"Score reduction function: `{score_reduction_func}`"
                f" not implemented. Must be one of {all_scoring_methods}"
            )
        )
    for tag_score, weight in zip(tag_scores, weights):
        cs_values = list(tag_score.values())
        score = scoring_method(cs_values, weight=weights, **reduction_func_arg)
        scoring.append(score)

    return scoring


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


@score_reduction_func
def mean_plus_weight(cs_values, weight, N, **kwargs):
    """Simple mean plus N * weights"""
    return (simple_mean(cs_values) + N * weight) / (N + 1)
