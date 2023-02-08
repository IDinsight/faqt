from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from faqt.model import Contextualisation, get_ordered_distance_matrix


@pytest.fixture(scope="module")
def faqs():
    full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    tagsets_data = yaml_dict["tags_refresh_data"]
    return [tagset_data for tagset_data in tagsets_data]


@pytest.fixture(scope="module")
def distance_matrix():

    full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    tagsets_data = yaml_dict["tags_refresh_data"]
    contexts = set(
        [tag for tagset_data in tagsets_data for tag in tagset_data["context"]]
    )
    contexts = list(contexts)
    distance_matrix = get_ordered_distance_matrix(contexts)
    return distance_matrix


class TestContextualisation:
    @pytest.mark.parametrize(
        "faq,context",
        [
            (
                [
                    {"faq_id": 1, "context": ["word", "music"]},
                    {"faq_id": 2, "context": ["holiday", "food"]},
                    {"faq_id": 3, "context": ["holiday", "music"]},
                ],
                ["word", "music", "holiday", "food"],
            ),
            (
                [
                    {"faq_id": 1, "context": ["jump", "run"]},
                    {"faq_id": 2, "context": ["shoot", "punch"]},
                    {"faq_id": 3, "context": ["jump", "sprint", "shoot"]},
                    {
                        "faq_id": 34,
                        "context": ["run", "punch", "sprint", "shoot", "jump"],
                    },
                ],
                ["run", "punch", "sprint", "shoot", "jump"],
            ),
        ],
    )
    def test_context_matrix_shape(self, faq, context):
        distance_matrix = get_ordered_distance_matrix(context_list=context)
        contextualisator = Contextualisation(faqs=faq, distance_matrix=distance_matrix)
        assert contextualisator._context_matrix.shape == (
            len(faq),
            len(distance_matrix.columns),
        )

    @pytest.mark.parametrize(
        "distance_matrix",
        [(pd.DataFrame())],
    )
    def test_empty_distance_matrix_return_error(self, faqs, distance_matrix):

        with pytest.raises(ValueError):
            contextualisator = Contextualisation(
                faqs=faqs, distance_matrix=distance_matrix
            )

    @pytest.mark.parametrize(
        "faq,context",
        [
            (
                [
                    {"faq_id": 1, "context": ["word", "music"]},
                    {"faq_id": 2, "context": ["holiday", "food"]},
                    {"faq_id": 3, "context": ["holiday", "music"]},
                ],
                ["word", "music", "holiday", "food"],
            ),
            (
                [
                    {"faq_id": 1, "context": ["jump", "run"]},
                    {"faq_id": 2, "context": ["shoot", "punch"]},
                    {"faq_id": 3, "context": ["jump", "sprint", "shoot"]},
                    {
                        "faq_id": 34,
                        "context": ["run", "punch", "sprint", "shoot", "jump"],
                    },
                ],
                ["run", "punch", "sprint", "shoot", "jump"],
            ),
        ],
    )
    def test_context_matrix_only_have_0_and_1(self, faq, context):
        distance_matrix = get_ordered_distance_matrix(context_list=context)
        contextualisator = Contextualisation(faqs=faq, distance_matrix=distance_matrix)
        assert sum(set(contextualisator._context_matrix.flatten())) == 1

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_empty_faqs_return_empty_list(self, distance_matrix):
        inbound_content = ["music"]
        contextualisator = Contextualisation(faqs=[], distance_matrix=distance_matrix)
        weights = contextualisator.get_context_weights(inbound_content)
        assert len(weights) == 0

    @pytest.mark.parametrize(
        "inbound_content",
        [([])],
    )
    def test_empty_content_throws_error(self, faqs, distance_matrix, inbound_content):
        contextualisator = Contextualisation(faqs=faqs, distance_matrix=distance_matrix)
        with pytest.raises(ValueError):
            weights = contextualisator.get_context_weights(inbound_content)

    @pytest.mark.parametrize(
        "inbound_content",
        [
            (["word", "music", "appreciation"]),
            (["holiday", "food", "word", "music", "musik"]),
        ],
    )
    def test_unknown_content_returns_error(
        self, faqs, distance_matrix, inbound_content
    ):
        contextualisator = Contextualisation(faqs=faqs, distance_matrix=distance_matrix)
        with pytest.raises(ValueError):
            weights = contextualisator.get_context_weights(inbound_content)

    @pytest.mark.parametrize(
        "inbound_content",
        [
            (["word", "music"]),
            (["holiday", "food"]),
        ],
    )
    def test_length_weights_vector(self, faqs, distance_matrix, inbound_content):

        contextualisator = Contextualisation(faqs=faqs, distance_matrix=distance_matrix)
        weights = contextualisator.get_context_weights(inbound_content)
        assert len(weights) == len(faqs)

    @pytest.mark.parametrize(
        "inbound_content",
        [
            (["word", "music"]),
            (["holiday", "food"]),
        ],
    )
    def test_weights_are_int_or_float(self, faqs, distance_matrix, inbound_content):
        contextualisator = Contextualisation(faqs=faqs, distance_matrix=distance_matrix)
        weights = contextualisator.get_context_weights(inbound_content)
        assert weights.dtype in (float, int)

    @pytest.mark.parametrize(
        "context_list",
        [
            (["morning", "night"]),
            (["breakfast", "lunch", "supper", "dinner"]),
        ],
    )
    def test_distance_matrix_is_square_matrix(self, context_list):
        distance_matrix = get_ordered_distance_matrix(context_list=context_list)
        size = len(context_list)
        assert distance_matrix.shape == (size, size)
