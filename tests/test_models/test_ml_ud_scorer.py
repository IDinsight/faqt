from functools import partial
from pathlib import Path

import pytest

from faqt.preprocessing import preprocess_text_for_keyword_rule
from faqt.model.urgency_detection.urgency_detection_base import MLBasedUD

from hunspell import Hunspell
from nltk.stem import PorterStemmer
import joblib


class TestMLBasedUD:
    @pytest.fixture
    def ml_model(self):
        full_path = Path(__file__).parents[1] / "data/ud_ml_models/model_test.joblib"
        return joblib.load(full_path)

    @pytest.fixture(scope="class")
    def preprocess_func(self):
        huns = Hunspell()
        stemmer = PorterStemmer()
        preprocess_func = partial(
            preprocess_text_for_keyword_rule,
            n_min_dashed_words_url=3,
            stem_func=stemmer.stem,
            spell_checker=huns,
            reincluded_stop_words=["what", "not", "how", "much", "where", "me"],
            ngram_min=1,
            ngram_max=2,
        )
        return preprocess_func

    def test_model_and_preprocessing_set_correctly(self, ml_model, preprocess_func):
        predictor = MLBasedUD(model=ml_model, preprocessor=preprocess_func)
        assert predictor.model == ml_model
        assert predictor.preprocessor == preprocess_func

    def test_get_model_returns_model(self, ml_model, preprocess_func):
        predictor = MLBasedUD(model=ml_model, preprocessor=preprocess_func)
        returned_model = predictor.get_model()
        assert ml_model == returned_model

    @pytest.mark.parametrize(
        "rule_id, message",
        [
            (
                0,
                "I have a headache, feel dizzy, and everything looks blurry",
            ),
            # True because it contains all included keywords
            (1, "My back pain is killing me :("),  # True, because it
            # includes all included keywords and no excluded keyword
            # True, because it doesn't have excluded keyword
        ],
    )
    def test_model_returns_prediction(
        self, ml_model, preprocess_func, rule_id, message
    ):
        predictor = MLBasedUD(model=ml_model, preprocessor=preprocess_func)
        assert isinstance(predictor.predict(message), float)
