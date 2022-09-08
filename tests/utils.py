import os
import shutil
from pathlib import Path

import boto3
import numpy as np
from gensim.models import KeyedVectors


def load_wv_pretrained_bin(folder, filename):
    """
    Load pretrained word2vec model from either local mount or S3
    based on environment var.

    TODO: make into a pure function and take ENV as input
    """

    if os.getenv("GITHUB_ACTIONS") == "true":
        bucket = os.getenv("WORD2VEC_BINARY_BUCKET")
        model = KeyedVectors.load_word2vec_format(
            f"s3://{bucket}/{filename}", binary=True
        )

    else:
        full_path = Path(__file__).parent / "data" / folder / filename
        model = KeyedVectors.load_word2vec_format(
            full_path,
            binary=True,
        )

    return model


def download_huggingface_model_from_s3(bucket, folder, filename):
    """Downloads model archive, unpackes it, then deletes the original
    archive. Returns the path to the huggingface model folder."""
    dir_path = Path(__file__).parent / "data" / folder
    full_path = dir_path / filename

    os.makedirs(dir_path, exist_ok=True)
    download_path = dir_path / f"{filename}.tar.gz"

    s3 = boto3.client("s3")
    s3.download_file(bucket, filename, download_path)

    shutil.unpack_archive(download_path, full_path)

    os.remove(download_path)

    return full_path


def get_topic_scores_for_message(inbound_vectors, topics, scores):
    """
    Returns tag_scores for the inbound vectors against each topic

    Parameters
    ----------
    inbound_vectors: List[Array]
        List of inbound tokens as word vectors
    topics: List[Topics]
        A list of Topic-like objects.
    scores: List[Dict]
        A list of CS tag_scores for each set of topic tags

    Returns
    -------
    Dict[int, float]
        A Dictionary with `_id` as key and similarity score as value
    """

    scoring = {}

    for topic, all_tag_scores in zip(topics, scores):
        cs_scores = list(all_tag_scores.values())
        scoring[topic._id] = (np.max(cs_scores) + np.mean(cs_scores)) / 2

    return scoring


def _filter_topic_scores_by_threshold(topic_scores, threshold):
    """

    Parameters
    ----------
    topic_scores : dict[int, float]
        With key as topic_id and value as tag_scores
    threshold : dict or float
        If dict: must have same keys as `topic_scores` and values as thresholds for
        each topic.
        If float: the same threshold is used for all topics

    Returns
    -------
    dict[int, float]
        With key as topic_id and value as tag_scores
    """

    if isinstance(threshold, dict):
        thresholded_scores = {}
        for topic, score in topic_scores.items():
            if threshold[topic] < score:
                thresholded_scores[topic] = score
    else:
        thresholded_scores = {
            topic: score for topic, score in topic_scores.items() if score > threshold
        }

    return thresholded_scores


def get_top_n_matches(scoring, n_top_matches):
    """
    Gives a list of tag_scores for each FAQ, return the top `n_top_matches` FAQs

    Parameters
    ----------
    scoring: Dict[int, Dict]
        Dict with faq_id as key and faq details and tag_scores as values.
        See return value of `get_faq_scores_for_message`.
    n_top_matches: int
        the number of top matches to return

    Returns
    -------
    List[Tuple(int, str)]
        A list of tuples of (faq_id, faq_content_to_send)._

    """
    matched_faq_titles = set()
    # Sort and copy over top matches
    top_matches_list = []
    for id in sorted(scoring, key=lambda x: scoring[x]["overall_score"], reverse=True):
        if scoring[id]["faq_title"] not in matched_faq_titles:
            top_matches_list.append(
                (scoring[id]["faq_title"], scoring[id]["faq_content_to_send"])
            )
            matched_faq_titles.add(scoring[id]["faq_title"])

        if len(matched_faq_titles) == n_top_matches:
            break
    return top_matches_list
