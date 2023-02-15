from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from faqt import Contextualization
from faqt.model.faq_matching.contextualization import get_ordered_distance_matrix


@pytest.fixture(scope="module")
def default_content_dict():
    full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    tagsets_data = yaml_dict["tags_refresh_data"]
    return {tagset_data["id"]: tagset_data["context"] for tagset_data in tagsets_data}


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
        "contents_dict,context",
        [
            (
                {
                    0: ["word", "music"],
                    1: ["beat", "album"],
                    2: ["beat", "music"],
                },
                ["word", "beat", "music", "album"],
            ),
            (
                {
                    0: ["jump", "run"],
                    1: ["shoot", "score"],
                    2: ["sprint", "jump", "shoot"],
                    3: ["run", "sprint", "jump", "shoot", "score"],
                },
                ["run", "sprint", "jump", "shoot", "score"],
            ),
        ],
    )
    def test_context_matrix_shape(self, contents_dict, context):
        distance_matrix = get_ordered_distance_matrix(context_list=context)
        contextualizer = Contextualization(
            contents_dict=contents_dict,
            distance_matrix=distance_matrix,
        )
        assert contextualizer._context_matrix.shape == (
            len(contents_dict),
            len(distance_matrix.columns),
        )

    def test_empty_distance_matrix_return_error(self, default_content_dict):
        distance_matrix = pd.DataFrame()
        with pytest.raises(AssertionError):
            contextualizer = Contextualization(
                contents_dict=default_content_dict,
                distance_matrix=distance_matrix,
            )

    @pytest.mark.parametrize(
        "contents_dict,context",
        [
            (
                {
                    0: ["word", "music"],
                    1: ["beat", "album"],
                    2: ["beat", "music"],
                },
                ["word", "beat", "music", "album"],
            ),
            (
                {
                    0: ["jump", "run"],
                    1: ["shoot", "score"],
                    2: ["sprint", "jump", "shoot"],
                    3: ["run", "sprint", "jump", "shoot", "score"],
                },
                ["run", "sprint", "jump", "shoot", "score"],
            ),
        ],
    )
    def test_context_matrix_only_have_0_and_1(self, contents_dict, context):
        distance_matrix = get_ordered_distance_matrix(context_list=context)
        contextualizer = Contextualization(
            contents_dict=contents_dict,
            distance_matrix=distance_matrix,
        )
        unique_values = set(contextualizer._context_matrix.flatten())
        assert len(unique_values - {0, 1}) == 0

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_empty_contents_return_empty_list(self, default_distance_matrix):
        message_context = ["code"]
        contextualizer = Contextualization(
            contents_dict=dict(), distance_matrix=default_distance_matrix
        )
        weights = contextualizer.get_context_weights(message_context)
        assert len(weights) == 0

    def test_empty_message_context_throws_error(
        self, default_content_dict, default_distance_matrix
    ):
        message_context = []
        contextualizer = Contextualization(
            contents_dict=default_content_dict,
            distance_matrix=default_distance_matrix,
        )
        with pytest.raises(ValueError):
            weights = contextualizer.get_context_weights(message_context)

    @pytest.mark.parametrize(
        "message_context",
        [
            (["design", "code ", "appreciation"]),
            (["design", "test", "deploy", "maintain", "musik"]),
        ],
    )
    def test_unknown_content_returns_error(
        self,
        default_content_dict,
        default_distance_matrix,
        message_context,
    ):
        contextualizer = Contextualization(
            contents_dict=default_content_dict,
            distance_matrix=default_distance_matrix,
        )
        with pytest.raises(ValueError):
            weights = contextualizer.get_context_weights(message_context)

    @pytest.mark.parametrize(
        "message_context",
        [
            (["deploy", "maintain"]),
            (["design", "deploy"]),
        ],
    )
    def test_length_weights_vector(
        self,
        default_content_dict,
        default_distance_matrix,
        message_context,
    ):

        contextualizer = Contextualization(
            contents_dict=default_content_dict,
            distance_matrix=default_distance_matrix,
        )
        weights = contextualizer.get_context_weights(message_context)
        assert len(weights.values()) == len(default_content_dict)

    @pytest.mark.parametrize(
        "message_context",
        [
            (["deploy", "maintain"]),
            (["design", "deploy"]),
        ],
    )
    def test_weights_are_int_or_float(
        self,
        default_content_dict,
        default_distance_matrix,
        message_context,
    ):
        contextualizer = Contextualization(
            contents_dict=default_content_dict,
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
