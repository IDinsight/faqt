from functools import partial
from pathlib import Path

import numpy as np
import pytest
import yaml
from faqt.model import Contextualisation


@pytest.fixture(scope="module")
def faqs():
    full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    tagsets_data = yaml_dict["tags_refresh_data"]
    return [tagset_data for tagset_data in tagsets_data]


@pytest.fixture(scope="module")
def contexts():

    full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    tagsets_data = yaml_dict["tags_refresh_data"]
    contexts = set(
        [tag for tagset_data in tagsets_data for tag in tagset_data["context"]]
    )
    return list(contexts)


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
    def test_context_matrix_size(self, faq, context):
        contextualisator = Contextualisation(faqs=faq, contexts=context)
        assert contextualisator._context_matrix.shape == (len(faq), len(context))

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_empty_faqs_return_empty_list(self):
        context = ["context1"]
        inbound_content = ["context1"]
        contextualisator = Contextualisation(faqs=[], contexts=context)
        weights = contextualisator.get_context_weights(inbound_content)
        assert len(weights) == 0

    @pytest.mark.parametrize(
        "inbound_content",
        [([])],
    )
    def test_empty_content_throws_error(self, faqs, contexts, inbound_content):
        contextualisator = Contextualisation(faqs=faqs, contexts=contexts)
        with pytest.raises(ValueError):
            weights = contextualisator.get_context_weights(inbound_content)

    @pytest.mark.parametrize(
        "inbound_content",
        [
            (["word", "music"]),
            (["holiday", "food"]),
        ],
    )
    def test_length_weights_vector(self, faqs, contexts, inbound_content):

        contextualisator = Contextualisation(faqs=faqs, contexts=contexts)
        weights = contextualisator.get_context_weights(inbound_content)
        assert len(weights) == len(faqs)
