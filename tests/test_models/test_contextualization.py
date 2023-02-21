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

    full_path = Path(__file__).parents[1] / "data/contextualization.yaml"
    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    contexts = yaml_dict["contexts_list"]

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

    @pytest.mark.parametrize(
        "contents_dict,context",
        [
            (
                {
                    0: ["word", "music"],
                    1: ["beat", "album"],
                    2: ["beat", "music", "single"],
                },
                ["word", "beat", "music", "album"],
            ),
            (
                {
                    0: ["jump", "run"],
                    1: ["shoot", "score"],
                    2: ["sprint", "jump", "shoot", "danse"],
                    3: ["run", "sprint", "jump", "shoot", "score"],
                },
                ["run", "sprint", "jump", "shoot", "score"],
            ),
        ],
    )
    def test_unknown_content_context_return_error(self, contents_dict, context):
        distance_matrix = get_ordered_distance_matrix(context_list=context)
        with pytest.raises(ValueError):
            contextualizer = Contextualization(
                contents_dict=contents_dict,
                distance_matrix=distance_matrix,
            )

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
    def test_unknown_message_context_returns_error(
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


class TestContextualizationAlgorithm:
    @pytest.fixture(scope="class")
    def contextualizer(self, default_distance_matrix, default_content_dict):
        return Contextualization(
            contents_dict=default_content_dict, distance_matrix=default_distance_matrix
        )

    @pytest.fixture(scope="class")
    def contextualizer_1(self, default_distance_matrix, default_content_dict):
        return Contextualization(
            contents_dict=default_content_dict,
            distance_matrix=default_distance_matrix,
            variance=1,
        )

    @pytest.fixture(scope="class")
    def contextualizer_10(self, default_distance_matrix, default_content_dict):
        return Contextualization(
            contents_dict=default_content_dict,
            distance_matrix=default_distance_matrix,
            variance=10,
        )

    @pytest.mark.parametrize(
        "message_context,expected_key_max,expected_key_min",
        [(["code", "test"], [1, 2, 4], 3), (["code", "maintain"], [2, 4, 6], 1)],
    )
    def test_two_context_in_message(
        self, contextualizer, message_context, expected_key_max, expected_key_min
    ):
        weights = contextualizer.get_context_weights(message_context)
        key_max = max(weights, key=weights.get)
        assert key_max == expected_key_max[0]
        assert weights[expected_key_min] < weights[expected_key_max[0]]
        if len(expected_key_max) > 1:
            assert (
                weights[expected_key_max[0]]
                == weights[expected_key_max[1]]
                == weights[expected_key_max[2]]
            )

    @pytest.mark.parametrize(
        "message_context,expected_key_min",
        [(["test"], 3), (["maintain"], 3), (["deploy"], 1)],
    )
    def test_contextualisation_variance(
        self,
        contextualizer,
        contextualizer_1,
        contextualizer_10,
        message_context,
        expected_key_min,
    ):
        weights_0_1 = contextualizer.get_context_weights(message_context)
        weights_1 = contextualizer_1.get_context_weights(message_context)
        weights_10 = contextualizer_10.get_context_weights(message_context)
        assert (
            max(weights_0_1, key=weights_0_1.get)
            == max(weights_1, key=weights_1.get)
            == max(weights_10, key=weights_10.get)
        )
        assert (
            weights_10[expected_key_min]
            < weights_1[expected_key_min]
            < weights_0_1[expected_key_min]
        )
