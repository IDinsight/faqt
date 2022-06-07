from pathlib import Path

import pytest
import yaml
from faqt.model import KeyedVectorsScorer
from faqt.preprocessing import preprocess_text_for_word_embedding
from dataclasses import dataclass
from typing import List

pytestmark = pytest.mark.slow


@dataclass
class Tagset:
    """
    Class for Tagset objects (to replicate functionality of SQLAlchemy ORM)
    """

    id: str
    title: str
    tags: List[str]
    content_to_send: str


class TestTagsetScorer:
    """Create a model with just the bare bones"""

    sample_messages = [
        "\U0001f600\U0001f929\U0001f617",
        """ I'm worried about the vaccines. Can I have some information? \U0001f600
            πλέων ἐπὶ οἴνοπα πόντον ἐπ᾽ ἀλλοθρόους ἀνθρώπους, ἐς Τεμέσην
        """,
        "cuoisn mircochippde my ddady with vcacines",
        "",
        """ The sun sets,
            In a silent sky,
            Echoing of a passing song,
            A long wait for the dawn,
            A record of time while it glides along.
        """,
    ]

    @pytest.fixture
    def basic_model(self, w2v_model):
        faqt = KeyedVectorsScorer(w2v_model)
        return faqt

    @pytest.fixture
    def hunspell_model(self, w2v_model, hunspell):
        faqt = KeyedVectorsScorer(
            w2v_model,
            hunspell=hunspell,
            tags_guiding_typos=["music", "food"],
        )
        return faqt

    @pytest.fixture(scope="class")
    def tagsets(self):
        full_path = Path(__file__).parent / "data/tag_test_data.yaml"
        with open(full_path) as file:
            yaml_dict = yaml.full_load(file)

        tagsets = [Tagset(**f) for f in yaml_dict["tags_refresh_data"]]
        return tagsets

    @pytest.mark.parametrize("input_text", sample_messages)
    def test_basic_model_score_with_empty_tags(self, basic_model, input_text):

        tokens = preprocess_text_for_word_embedding(input_text, {}, 0)
        basic_model.set_tags([])
        assert len(basic_model.tagset) == 0

        scores, _ = basic_model.score(tokens)

        assert sum(any(i.values()) for i in scores) == 0

    @pytest.mark.parametrize(
        "input_text",
        sample_messages,
    )
    def test_basic_model_score_with_nonempty_tags(
        self, hunspell_model, tagsets, input_text
    ):

        hunspell_model.set_tags([tagset.tags for tagset in tagsets])
        assert len(hunspell_model.tagset) == len(tagsets)
        tokens = preprocess_text_for_word_embedding(input_text, {}, 0)

        scores, _ = hunspell_model.score(tokens)

        if len(tokens) == 0:
            assert sum(any(i.values()) for i in scores) == 0
        else:
            assert sum(any(i.values()) for i in scores) == len(tagsets)
