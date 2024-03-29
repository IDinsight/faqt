from functools import partial
from pathlib import Path
import joblib
import pytest
from faqt.model.urgency_detection.urgency_detection_base import (
    KeywordRule,
    evaluate_keyword_rule,
    RuleBasedUD,
    MLBasedUD,
)
from faqt.preprocessing import preprocess_text_for_keyword_rule
from hunspell import Hunspell
from nltk.stem import PorterStemmer


class TestKeywordRule:
    def test_keyword_rule_empty_raises_error(self):
        with pytest.raises(ValueError):
            KeywordRule()

    def test_keyword_rule_duplicate_raises_warning(self):
        with pytest.raises(Warning):
            KeywordRule(include=["hat", "hit"], exclude=["hut", "hat"])

    def test_keyword_rule_default_attributes(self):
        only_include_rule = KeywordRule(include=[""])
        assert isinstance(only_include_rule.exclude, list)
        assert len(only_include_rule.exclude) == 0

        only_exclude_rule = KeywordRule(exclude=[""])
        assert isinstance(only_exclude_rule.include, list)
        assert len(only_exclude_rule.include) == 0

    def test_keyword_rule_attributes_set_correctly(self):
        rule = KeywordRule(include=["a", "b", "c"], exclude=["x", "y", "z"])
        assert rule.include == ["a", "b", "c"]
        assert rule.exclude == ["x", "y", "z"]


class TestKeywordRuleEvaluation:
    @pytest.fixture
    def keyword_rules(self):
        rules = [
            KeywordRule(
                include=["headach", "dizzi", "blurri"],
            ),
            KeywordRule(include=["kill_me", "pain"], exclude=["painkiller"]),
            KeywordRule(exclude=["hello"]),
        ]
        return rules

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

    @pytest.mark.parametrize(
        "rule_id, message",
        [
            (0, "I have a headache"),
            (0, "Please STOP messaging me 😭😭"),
            (1, "Please STOP messaging me 😭😭"),
        ],
    )
    def test_rule_false_on_absence_of_include(
        self, preprocess_func, keyword_rules, rule_id, message
    ):
        msg = preprocess_func(message)
        assert evaluate_keyword_rule(msg, keyword_rules[rule_id]) is False

    @pytest.mark.parametrize(
        "rule_id, message",
        [
            (1, "Please wht kind of painkillers would u recomend me"),
            (2, "hello wht kind of painkillers would u recomend me"),
        ],
    )
    def test_rule_false_on_presence_of_excluded_keyword(
        self,
        preprocess_func,
        keyword_rules,
        rule_id,
        message,
    ):
        msg = preprocess_func(message)
        assert evaluate_keyword_rule(msg, keyword_rules[rule_id]) is False

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
            (2, "hi hi"),  # True, because it doesn't have excluded keyword
        ],
    )
    def test_rule_true(self, preprocess_func, keyword_rules, rule_id, message):
        msg = preprocess_func(message)
        assert evaluate_keyword_rule(msg, keyword_rules[rule_id]) is True

    def test_rules_and_preprocessing_set_correctly(
        self, keyword_rules, preprocess_func
    ):
        predictor = RuleBasedUD(model=keyword_rules, preprocessor=preprocess_func)
        assert (
            predictor.model == keyword_rules
            and predictor.preprocessor == preprocess_func
        )

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
            (2, "hi hi"),  # True, because it doesn't have excluded keyword
        ],
    )
    def test_rule_based_ud_predict_score_return_array(
        self, keyword_rules, preprocess_func, rule_id, message
    ):
        predictor = RuleBasedUD(model=keyword_rules, preprocessor=preprocess_func)
        scores = predictor.predict_scores(message)
        assert isinstance(scores, list)
        assert len(scores) == len(keyword_rules)

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
            (2, "hi hi"),  # True, because it doesn't have excluded keyword
        ],
    )
    def test_rule_based_ud_predict_return_float(
        self, keyword_rules, preprocess_func, rule_id, message
    ):
        predictor = RuleBasedUD(model=keyword_rules, preprocessor=preprocess_func)
        scores = predictor.predict(message)
        assert isinstance(scores, float)

    @pytest.mark.parametrize(
        "rule_id, message, expected",
        [
            (
                0,
                "Hello I have a headache, feel dizzy, and everything looks " "blurry",
                1.0,
            ),
            (
                1,
                "Hello I have a headache, feel dizzy, and everything looks " "blurry",
                0.0,
            ),
            (
                2,
                "Hello I have a headache, feel dizzy, and everything looks " "blurry",
                0.0,
            ),
            (0, "hello My back pain is killing me :(", 0.0),
            (1, "hello My back pain is killing me :(", 1.0),
            (2, "hello My back pain is killing me :(", 0.0),
            (0, "What painkiller shd i take", 0.0),
            (1, "What painkiller shd i take", 0.0),
            (2, "What painkiller shd i take", 1.0),
        ],
    )
    def test_evaluate_rules_true_for_only_one_rule(
        self, preprocess_func, keyword_rules, rule_id, message, expected
    ):
        keyword_model = RuleBasedUD(model=keyword_rules, preprocessor=preprocess_func)
        vals = keyword_model.predict_scores(message)
        assert vals[rule_id] == expected

    @pytest.mark.parametrize(
        "message",
        [
            "hello Please wht kind of painkillers would u recomend me",
            "Please STOP messaging me 😭😭hello",
        ],
    )
    def test_evaluate_rules_all_false_for_excluded_kw_and_no_included_kw(
        self, preprocess_func, keyword_rules, message
    ):
        keyword_model = RuleBasedUD(model=keyword_rules, preprocessor=preprocess_func)
        vals = keyword_model.predict_scores(message)
        assert all(result == 0.0 for result in vals)
