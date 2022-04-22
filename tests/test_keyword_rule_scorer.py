from hunspell import Hunspell
import pytest
from functools import partial
from nltk.stem import PorterStemmer

from faqt.model.keyword_rule_matching_model import (
    evaluate_keyword_rule,
    evaluate_keyword_rules,
    KeywordRule
)
from faqt.preprocessing import preprocess_text_for_keyword_rule


class TestKeywordRule:
    def test_keyword_rule_empty_raises_error(self):
        with pytest.raises(ValueError):
            KeywordRule()

    def test_keyword_rule_default_attributes(self):
        only_include_rule = KeywordRule(include=[''])
        assert isinstance(only_include_rule.exclude, list)
        assert len(only_include_rule.exclude) == 0

        only_exclude_rule = KeywordRule(exclude=[''])
        assert isinstance(only_exclude_rule.include, list)
        assert len(only_exclude_rule.include) == 0

    def test_keyword_rule_attributes_set_correctly(self):
        rule = KeywordRule(include=['a', 'b', 'c'], exclude=['x', 'y', 'z'])
        assert rule.include == ['a', 'b', 'c']
        assert rule.exclude == ['x', 'y', 'z']


class TestKeywordRuleEvaluation:
    @pytest.fixture
    def keyword_rules(self):
        rules = [
            KeywordRule(include=['headach', 'dizzi', 'blurri'],),
            KeywordRule(include=['kill_me', 'pain'], exclude=['painkiller']),
        ]
        return rules

    @pytest.fixture(scope="class")
    def preprocess_func(self):
        huns = Hunspell()
        stemmer = PorterStemmer()
        preprocess_func = partial(
            preprocess_text_for_keyword_rule,
            n_min_dashed_words_url=3, stem_func=stemmer.stem,
            spell_checker=huns,
            reincluded_stop_words=['what', 'not', 'how', 'much', 'where', 'me'],
            ngram_min=1, ngram_max=2,
        )
        return preprocess_func

    def test_rule_false_on_absence_of_include(self, preprocess_func,
                                              keyword_rules):
        msg = preprocess_func("I have a headache")
        assert not evaluate_keyword_rule(msg, keyword_rules[0])

        msg = preprocess_func("Please STOP messaging me 😭😭")
        assert not evaluate_keyword_rule(msg, keyword_rules[0])
        assert not evaluate_keyword_rule(msg, keyword_rules[1])

    def test_rule_false_on_presence_of_exclude(
            self, preprocess_func, keyword_rules
    ):
        msg = preprocess_func(
            "Please wht kind of painkillers would u recomend me"
        )
        assert not evaluate_keyword_rule(msg, keyword_rules[1])

    def test_rule_true(self, preprocess_func, keyword_rules):
        msg = preprocess_func(
            "I have a headache, feel dizzy, and everything looks blurry")
        assert evaluate_keyword_rule(msg, keyword_rules[0])

        msg = preprocess_func(
            "My back pain is killing me :(")
        assert evaluate_keyword_rule(msg, keyword_rules[1])

    def test_rules_true_for_one(self, preprocess_func, keyword_rules):
        msg = preprocess_func(
            "I have a headache, feel dizzy, and everything looks blurry"
        )
        vals = evaluate_keyword_rules(msg, keyword_rules)
        assert vals[0]
        assert not vals[1]

        msg = preprocess_func("My back pain is killing me :(")
        vals = evaluate_keyword_rules(msg, keyword_rules)
        assert not vals[0]
        assert vals[1]

    def test_rules_false_for_none(self, preprocess_func, keyword_rules):
        msg = preprocess_func("Please wht kind of painkillers would u recomend me")
        vals = evaluate_keyword_rules(msg, keyword_rules)
        assert not vals[0]
        assert not vals[1]

        msg = preprocess_func("Please STOP messaging me 😭😭")
        vals = evaluate_keyword_rules(msg, keyword_rules)
        assert not vals[0]
        assert not vals[1]
