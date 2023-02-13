from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from faqt import Contextualization
from faqt.model.faq_matching.contextualization import get_ordered_distance_matrix


@pytest.fixture(scope="module")
def default_contents():
    full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    tagsets_data = yaml_dict["tags_refresh_data"]
    return [tagset_data["context"] for tagset_data in tagsets_data]


@pytest.fixture(scope="module")
def default_distance_matrix():

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


@pytest.fixture(scope="module")
def default_contents_id():
    full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    tagsets_data = yaml_dict["tags_refresh_data"]
    return [tagset_data["id"] for tagset_data in tagsets_data]


class TestContextualization:
    @pytest.mark.parametrize(
        "content_id,content,context",
        [
            (
                ["content_1", "content_2", "content_3"],
                [
                    ["word", "music"],
                    ["holiday", "food"],
                    ["holiday", "music"],
                ],
                ["word", "music", "holiday", "food"],
            ),
            (
                ["content_1", "content_2", "content_3", "content_4"],
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
    def test_context_matrix_shape(self, content_id, content, context):
        distance_matrix = get_ordered_distance_matrix(context_list=context)
        contextualizer = Contextualization(
            contents_id=content_id,
            contents_context=content,
            distance_matrix=distance_matrix,
        )
        assert contextualizer._context_matrix.shape == (
            len(content),
            len(distance_matrix.columns),
        )

    def test_empty_distance_matrix_return_error(
        self, default_contents, default_contents_id
    ):
        distance_matrix = pd.DataFrame()
        with pytest.raises(ValueError):
            contextualizer = Contextualization(
                contents_id=default_contents_id,
                contents_context=default_contents,
                distance_matrix=distance_matrix,
            )

    @pytest.mark.parametrize(
        "content_id,content,context",
        [
            (
                ["content_1", "content_2", "content_3"],
                [
                    ["word", "music"],
                    ["holiday", "food"],
                    ["holiday", "music"],
                ],
                ["word", "music", "holiday", "food"],
            ),
            (
                ["content_1", "content_2", "content_3", "content_4"],
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
    def test_context_matrix_only_have_0_and_1(self, content_id, content, context):
        distance_matrix = get_ordered_distance_matrix(context_list=context)
        contextualizer = Contextualization(
            contents_id=content_id,
            contents_context=content,
            distance_matrix=distance_matrix,
        )
        unique_values = set(contextualizer._context_matrix.flatten())
        assert len(unique_values - {0, 1}) == 0

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_empty_contents_return_empty_list(self, default_distance_matrix):
        message_context = ["music"]
        contextualizer = Contextualization(
            contents_id=[], contents_context=[], distance_matrix=default_distance_matrix
        )
        weights = contextualizer.get_context_weights(message_context)
        assert len(weights) == 0

    def test_empty_message_context_throws_error(
        self, default_contents_id, default_contents, default_distance_matrix
    ):
        message_context = []
        contextualizer = Contextualization(
            contents_id=default_contents_id,
            contents_context=default_contents,
            distance_matrix=default_distance_matrix,
        )
        with pytest.raises(ValueError):
            weights = contextualizer.get_context_weights(message_context)

    @pytest.mark.parametrize(
        "message_context",
        [
            (["word", "music", "appreciation"]),
            (["holiday", "food", "word", "music", "musik"]),
        ],
    )
    def test_unknown_content_returns_error(
        self,
        default_contents_id,
        default_contents,
        default_distance_matrix,
        message_context,
    ):
        contextualizer = Contextualization(
            contents_id=default_contents_id,
            contents_context=default_contents,
            distance_matrix=default_distance_matrix,
        )
        with pytest.raises(ValueError):
            weights = contextualizer.get_context_weights(message_context)

    @pytest.mark.parametrize(
        "message_context",
        [
            (["word", "music"]),
            (["holiday", "food"]),
        ],
    )
    def test_length_weights_vector(
        self,
        default_contents_id,
        default_contents,
        default_distance_matrix,
        message_context,
    ):

        contextualizer = Contextualization(
            contents_id=default_contents_id,
            contents_context=default_contents,
            distance_matrix=default_distance_matrix,
        )
        weights = contextualizer.get_context_weights(message_context)
        assert len(weights.values()) == len(default_contents)

    @pytest.mark.parametrize(
        "message_context",
        [
            (["word", "music"]),
            (["holiday", "food"]),
        ],
    )
    def test_weights_are_int_or_float(
        self,
        default_contents_id,
        default_contents,
        default_distance_matrix,
        message_context,
    ):
        contextualizer = Contextualization(
            contents_id=default_contents_id,
            contents_context=default_contents,
            distance_matrix=default_distance_matrix,
        )
        weights = contextualizer.get_context_weights(message_context)
        assert np.array(list(weights.values())).dtype in (float, int)

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
