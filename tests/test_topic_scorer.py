from pathlib import Path

import pytest
import yaml
from faqt.preprocessing import preprocess_text
from faqt.model import TopicModelScorer
from dataclasses import dataclass
from typing import List

pytestmark = pytest.mark.slow


@dataclass
class Topic:
    """
    Class for FAQ objects (to replicate functionality of SQLAlchemy ORM)
    """

    _id: str
    tags: List[str]


class TestTopicModelScorer:
    sample_messages = [
        "\U0001f600\U0001f929\U0001f617",
        """ I'm worried about the vaccines. Can I have some information? \U0001f600
            πλέων ἐπὶ οἴνοπα πόντον ἐπ᾽ ἀλλοθρόους ἀνθρώπους, ἐς Τεμέσην
        """,
        "cuoisn mircochippde my ddady with vcacines + correct spelling",
        "",
        """ The sun sets,
            In a silent sky,
            Echoing of a passing song,
            A long wait for the dawn,
            A record of time while it glides along.
        """,
        """ Me: You care deeply for her don't you?
            Him: I've always cared for her but now I care more, she's my rib
            Me: I'm glad.. Thank you very much

            To be continued
        """,
        "Blinkdrink is a made up word",
    ]

    @pytest.fixture
    def basic_model(self, w2v_model):
        model = TopicModelScorer(w2v_model)
        return model

    @pytest.fixture
    def extra_words_model(self, w2v_model, hunspell):
        model = TopicModelScorer(
            w2v_model, glossary={"blinkdrink": {"blink": 0.5, "drunk": 0.5}}
        )
        return model

    @pytest.fixture(scope="class")
    def topics(self):
        full_path = Path(__file__).parent / "data/topics_data.yaml"
        with open(full_path) as file:
            yaml_dict = yaml.full_load(file)

        topics = [Topic(**f) for f in yaml_dict["topic_data"]]
        return topics

    @pytest.fixture(scope="class")
    def topics_novocab_tags(self):
        """Return topics with some tags that are no in vocab"""
        full_path = Path(__file__).parent / "data/topics_data.yaml"
        with open(full_path) as file:
            yaml_dict = yaml.full_load(file)

        topics = [Topic(**f) for f in yaml_dict["topic_data_nonvocab_tags"]]
        return topics

    @pytest.mark.parametrize("input_text", sample_messages)
    def test_basic_model_score_with_empty_topics(self, basic_model, input_text):
        tokens = preprocess_text(input_text, {}, 0)
        basic_model.fit([])
        assert len(basic_model.topics) == 0

        matches, a = basic_model.score(tokens, k=10, floor=1)
        assert bool(matches) is False

    @pytest.mark.parametrize(
        "input_text",
        sample_messages,
    )
    def test_basic_model_score_with_topics(self, basic_model, topics, input_text):

        basic_model.fit(topics)
        assert len(basic_model.topics) == len(topics)

        tokens = preprocess_text(input_text, {}, 0)
        matches, _ = basic_model.score(tokens, k=10, floor=1)
        expected_bool = len(tokens) != 0
        assert bool(matches) is expected_bool

    @pytest.mark.parametrize("input_text", sample_messages)
    def test_glossary_improves_score(
        self, basic_model, extra_words_model, topics, input_text
    ):
        basic_model.fit(topics)
        extra_words_model.fit(topics)

        tokens = preprocess_text(input_text, {}, 0)

        scores_basic, _ = basic_model.score(tokens, k=10, floor=1)
        scores_glossary, _ = extra_words_model.score(tokens, k=10, floor=1)

        if "Blinkdrink" in input_text:
            assert sum(scores_basic.values()) < sum(scores_glossary.values())
        else:
            assert sum(scores_basic.values()) == sum(scores_glossary.values())

    def test_warning_if_tags_not_in_vocab(self, basic_model, topics_novocab_tags):
        with pytest.warns(RuntimeWarning):
            basic_model.fit(topics_novocab_tags)
