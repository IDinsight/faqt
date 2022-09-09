from pathlib import Path

import numpy as np
import pytest
import yaml
from faqt.model import QuestionAnswerBERTScorer
from transformers import Pipeline


def load_test_data():
    full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"

    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    contents_data = yaml_dict["tags_refresh_data"]
    return [
        (question, i)
        for i, d in enumerate(contents_data)
        for question in d["example_questions"]
    ]


class TestQuestionAnswerBERTScorer:
    """Test QuestionAnswerBERTScorer class"""

    test_data = load_test_data()

    @pytest.fixture
    def bert_scorer(self):
        folder = "sequence_classification_models"
        model_folder = "huggingface_model"
        full_path = Path(__file__).parent / "data" / folder / model_folder
        path = str(full_path)

        return QuestionAnswerBERTScorer(model_path=path)

    @pytest.fixture(scope="class")
    def contents(self):
        full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"

        with open(full_path) as file:
            yaml_dict = yaml.full_load(file)

        contents_data = yaml_dict["tags_refresh_data"]
        return [d["content_to_send"] for d in contents_data]

    def test_scorer_model_is_loaded(self, bert_scorer):
        assert isinstance(bert_scorer.model, Pipeline)

    def test_scoring_without_setting_raises_error(self, bert_scorer):
        with pytest.raises(
            ValueError, match="Set contents first using `self\.set_contents\(\)`."
        ):
            bert_scorer.score_contents("test message")

    def test_setting_contents_saves_contents(self, bert_scorer, contents):
        set_scorer = bert_scorer.set_contents(contents=contents)

        assert all(c1 == c2 for c1, c2 in zip(set_scorer.contents, contents))

    @pytest.mark.parametrize("question, correct_content_idx", test_data)
    def test_score_contents(self, bert_scorer, question, correct_content_idx, contents):
        bert_scorer.set_contents(contents=contents)
        scores = bert_scorer.score_contents(question)

        assert np.argmax(scores) == correct_content_idx

    def test_score_contetns_on_empty_msg_still_returns_scores(
        self, bert_scorer, contents
    ):
        bert_scorer.set_contents(contents=contents)
        scores = bert_scorer.score_contents("")

        assert len(scores) == len(contents)

    def test_score_contetns_on_long_msg_still_returns_scores(
        self, bert_scorer, contents
    ):
        """Checks that inputs with length > model max length gets truncated"""
        bert_scorer.set_contents(contents=contents)
        num_tokens = bert_scorer.model.tokenizer.model_max_length + 1
        really_long_message = "word " * num_tokens
        scores = bert_scorer.score_contents(really_long_message)

        assert len(scores) == len(contents)
