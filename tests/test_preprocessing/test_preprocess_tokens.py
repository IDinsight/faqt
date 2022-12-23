import sys

import pytest
from faqt.preprocessing.tokens import (
    CustomHunspell,
    connect_phrases,
    get_ngrams,
    remove_stop_words,
)

pytestmark = pytest.mark.fast


class TestNgramGeneration:
    def test_ngram_generation(self):
        res = get_ngrams(["a"], 1, 2)
        assert res[0][0] == "a"
        assert len(res) == 1

        res = get_ngrams(["a", "b", "c"], 1, 2)
        expected_result = [("a",), ("b",), ("c",), ("a", "b"), ("b", "c")]
        for x, y in zip(res, expected_result):
            assert x == y

    def test_ngram_length_1(self):
        res = get_ngrams(["a"], 2, 2)
        assert len(res) == 0

    def test_bad_ngram_range(self):
        with pytest.raises(AssertionError):
            get_ngrams(["a", "b", "c"], -1, 2)

        with pytest.raises(AssertionError):
            get_ngrams(["a", "b", "c"], 1, -1)

        with pytest.raises(AssertionError):
            get_ngrams(["a", "b", "c"], 2, 1)


class TestIncludingStopWords:
    @pytest.fixture
    def sentence(self):
        return ["Who", "let", "the", "dogs", "out"]

    @pytest.mark.parametrize(
        "stop_words_to_add_back, expected",
        [
            ([""], ["let", "dogs"]),
            (["who"], ["Who", "let", "dogs"]),
            (["who", "the"], ["Who", "let", "the", "dogs"]),
            (["the", "out"], ["let", "the", "dogs", "out"]),
        ],
    )
    def test_reincluded_stop_word(self, sentence, stop_words_to_add_back, expected):
        """
        Testing if adding back stop words function works
        """
        assert (
            remove_stop_words(sentence, reincluded_stop_words=stop_words_to_add_back)
            == expected
        )


class TestEntityRecognition:
    def test_entity_recognition_simple(self):
        """
        Test parsing out of entities using simple dictionary
        """
        simple_dict = {("african", "union"): "African_Union"}

        various_cases = [
            [
                "african",
                "union",
            ],
            [
                "African",
                "union",
            ],
            [
                "african",
                "Union",
            ],
            [
                "AFRICAN",
                "UNION",
            ],
            [
                "aFrican",
                "union",
            ],
            [
                "aFRICAN",
                "UNion",
            ],
        ]

        for msg in various_cases:
            assert connect_phrases(msg, simple_dict) == ["African_Union"]

    def test_entity_recognition_in_sentence(self):
        """
        Test parsing out of entities from sentence, using simple dictionary
        """
        simple_dict = {("african", "union"): "African_Union"}

        desired_results = [
            (
                ["african", "union", "leadership", "structure"],
                [
                    "African_Union",
                    "leadership",
                    "structure",
                ],
            ),
            (
                ["is", "the", "african", "union", "based", "in", "Addis"],
                [
                    "is",
                    "the",
                    "African_Union",
                    "based",
                    "in",
                    "Addis",
                ],
            ),
            (
                ["how", "many", "countries", "are", "in", "the", "African", "UNION"],
                [
                    "how",
                    "many",
                    "countries",
                    "are",
                    "in",
                    "the",
                    "African_Union",
                ],
            ),
            (["african", "union", "size"], ["African_Union", "size"]),
            (["history", "AfricAn", "unIon"], ["history", "African_Union"]),
        ]

        for msg, desired_result in desired_results:
            assert connect_phrases(msg, simple_dict) == desired_result


class TestCustomHunspell:
    def test_importing_faqt_without_optional_package_succeeds(self):
        # Make transformers library not available
        sys.modules["hunspell"] = None

        try:
            import faqt
        except ImportError:
            pytest.fail(
                "Unexpected ImportError: faqt import should work even if the optional library `hunspell` is not installed."
            )

    def test_import_error_raised_if_optional_package_unavailable(self):
        # Patch the module-level global variable that's set during import
        import faqt

        faqt.preprocessing.tokens._has_hunspell = False

        with pytest.raises(ImportError, match=r"Could not import hunspell library. "):
            faqt.preprocessing.tokens.CustomHunspell()

    def test_custom_spell_check(self):
        huns = CustomHunspell(custom_spell_check_list=["texting"])
        assert huns.spell("texting")
        assert not huns.spell("texter")

        vanilla_huns = CustomHunspell()
        assert not vanilla_huns.spell("texting")

    def test_custom_spell_correct_map(self):
        spell_correct_map = {"jondis": "jaundice", "pewking": "puking"}
        huns = CustomHunspell(custom_spell_correct_map=spell_correct_map)

        assert huns.suggest("jondis") == ("jaundice",)
        assert huns.suggest("pewking") == ("puking",)

    def test_priority_words(self):
        huns = CustomHunspell(priority_words=["pain", "child"])

        assert huns.suggest("pian") == ("pain",)  # not piano
        assert huns.suggest("chils") == ("child",)  # not chills

    def test_custom_spell_correct_overrides_priority_word(self):
        spell_correct_map = {
            "chils": "chills",
        }
        huns = CustomHunspell(
            custom_spell_correct_map=spell_correct_map, priority_words=["pain", "child"]
        )

        assert huns.suggest("chils") == ("chills",)
