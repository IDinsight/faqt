from pathlib import Path

import pytest
import yaml
from faqt.preprocessing import preprocess_text
from faqt.model import FAQScorer
from dataclasses import dataclass
from typing import List

pytestmark = pytest.mark.slow


@dataclass
class FAQ:
    """
    Class for FAQ objects (to replicate functionality of SQLAlchemy ORM)
    """

    faq_id: str
    faq_title: str
    faq_tags: List[str]
    faq_content_to_send: str


class TestFAQScorer:
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
        faqt = FAQScorer(w2v_model)
        return faqt

    @pytest.fixture
    def hunspell_model(self, w2v_model, hunspell):
        faqt = FAQScorer(
            w2v_model, hunspell=hunspell, tags_guiding_typos=["music", "food"]
        )
        return faqt

    @pytest.fixture(scope="class")
    def faqs(self):
        full_path = Path(__file__).parent / "data/faq_data.yaml"
        with open(full_path) as file:
            yaml_dict = yaml.full_load(file)

        faqs = [FAQ(**f) for f in yaml_dict["faq_refresh_data"]]
        return faqs

    @pytest.mark.parametrize("input_text", sample_messages)
    def test_basic_model_score_with_empty_faq(self, basic_model, input_text):
        tokens = preprocess_text(input_text, {}, 0)
        basic_model.fit([])
        assert len(basic_model.faqs) == 0

        matches, a, b = basic_model.score(tokens, n_top_matches=3)
        assert bool(matches) is False

    @pytest.mark.parametrize(
        "input_text",
        sample_messages,
    )
    def test_basic_model_score_with_faqs(self, hunspell_model, faqs, input_text):
        hunspell_model.fit(faqs)
        assert len(hunspell_model.faqs) == len(faqs)
        tokens = preprocess_text(input_text, {}, 0)

        matches, a, b = hunspell_model.score(tokens, n_top_matches=3, k=10, floor=1)
        expected_bool = len(tokens) != 0
        assert bool(matches) is expected_bool
