from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from faqt import Contextualization
from faqt.model.faq_matching.contextualization import get_ordered_distance_matrix


@pytest.fixture(scope="module")
def contents():
    full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    tagsets_data = yaml_dict["tags_refresh_data"]
    return [tagset_data["context"] for tagset_data in tagsets_data]


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


class TestContextualization:
    @pytest.mark.parametrize(
        "content,context",
        [
            (
                [
                    ["word", "music"],
                    ["holiday", "food"],
                    ["holiday", "music"],
                ],
                ["word", "music", "holiday", "food"],
            ),
            (
                [
                    ["jump", "run"],
                    ["shoot", "punch"],
                    ["jump", "sprint", "shoot"],
                    ["run", "punch", "sprint", "shoot", "jump"],
                ],
                ["run", "punch", "sprint", "shoot", "jump"],
            ),
        ],
    )
    def test_context_matrix_shape(self, content, context):
        distance_matrix = get_ordered_distance_matrix(context_list=context)
        contextualizator = Contextualization(
            contents=content, distance_matrix=distance_matrix
        )
        assert contextualizator._context_matrix.shape == (
            len(content),
            len(distance_matrix.columns),
        )

    @pytest.mark.parametrize(
        "distance_matrix",
        [(pd.DataFrame())],
    )
    def test_empty_distance_matrix_return_error(self, contents, distance_matrix):

        with pytest.raises(ValueError):
            contextualizator = Contextualization(
                contents=contents, distance_matrix=distance_matrix
            )

    @pytest.mark.parametrize(
        "content,context",
        [
            (
                [
                    ["word", "music"],
                    ["holiday", "food"],
                    ["holiday", "music"],
                ],
                ["word", "music", "holiday", "food"],
            ),
            (
                [
                    ["jump", "run"],
                    ["shoot", "punch"],
                    ["jump", "sprint", "shoot"],
                    ["run", "punch", "sprint", "shoot", "jump"],
                ],
                ["run", "punch", "sprint", "shoot", "jump"],
            ),
        ],
    )
    def test_context_matrix_only_have_0_and_1(self, content, context):
        distance_matrix = get_ordered_distance_matrix(context_list=context)
        contextualizator = Contextualization(
            contents=content, distance_matrix=distance_matrix
        )
        assert sum(set(contextualizator._context_matrix.flatten())) == 1

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_empty_contents_return_empty_list(self, distance_matrix):
        message_context = ["music"]
        contextualizator = Contextualization(
            contents=[], distance_matrix=distance_matrix
        )
        weights = contextualizator.get_context_weights(message_context)
        assert len(weights) == 0

    @pytest.mark.parametrize(
        "message_context",
        [([])],
    )
    def test_empty_content_throws_error(
        self, contents, distance_matrix, message_context
    ):
        contextualizator = Contextualization(
            contents=contents, distance_matrix=distance_matrix
        )
        with pytest.raises(ValueError):
            weights = contextualizator.get_context_weights(message_context)

    @pytest.mark.parametrize(
        "message_context",
        [
            (["word", "music", "appreciation"]),
            (["holiday", "food", "word", "music", "musik"]),
        ],
    )
    def test_unknown_content_returns_error(
        self, contents, distance_matrix, message_context
    ):
        contextualizator = Contextualization(
            contents=contents, distance_matrix=distance_matrix
        )
        with pytest.raises(ValueError):
            weights = contextualizator.get_context_weights(message_context)

    @pytest.mark.parametrize(
        "message_context",
        [
            (["word", "music"]),
            (["holiday", "food"]),
        ],
    )
    def test_length_weights_vector(self, contents, distance_matrix, message_context):

        contextualizator = Contextualization(
            contents=contents, distance_matrix=distance_matrix
        )
        weights = contextualizator.get_context_weights(message_context)
        assert len(weights) == len(contents)

    @pytest.mark.parametrize(
        "message_context",
        [
            (["word", "music"]),
            (["holiday", "food"]),
        ],
    )
    def test_weights_are_int_or_float(self, contents, distance_matrix, message_context):
        contextualizator = Contextualization(
            contents=contents, distance_matrix=distance_matrix
        )
        weights = contextualizator.get_context_weights(message_context)
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
