import pytest
from faqt.preprocessing import (
    preprocess_text_for_keyword_rule,
    preprocess_text_for_word_embedding,
)
from faqt.preprocessing.tokens import CustomHunspell
from hunspell import Hunspell
from nltk.stem import PorterStemmer

pytestmark = pytest.mark.fast


class TestPreprocessingForWordEmbedding:
    def test_punctuation_tokenization_stopwords(self):
        """
        Test removal of punctuation, tokenization, and removal of stopwords
        """
        desired_results = {
            "It was the BEST of times // it was the wORST of Times // it was "
            "the age "
            "of wIsDoM // it was the age of foolishness!!": [
                "BEST",
                "times",
                "wORST",
                "Times",
                "age",
                "wIsDoM",
                "age",
                "foolishness",
            ],
            "'Shoot all the bluejays you want---if you can hit 'em, "
            "but remember it's "
            "a sin to kill_a_mockingbird.'": [
                "Shoot",
                "bluejays",
                "want",
                "hit",
                "em",
                "remember",
                "sin",
                "kill",
                "mockingbird",
            ],
            "Nobody, of the hundreds of people that had visited the Fair, "
            "knew that a "
            "grey spider had played the most important part of all.": [
                "Nobody",
                "hundreds",
                "people",
                "visited",
                "Fair",
                "knew",
                "grey",
                "spider",
                "played",
                "important",
                "part",
            ],
            "So it goes...": ["goes"],
        }

        for msg in desired_results:
            assert (
                preprocess_text_for_word_embedding(msg, {}, 0) == desired_results[msg]
            )

    @pytest.fixture
    def sentence(self):
        return "Who let the dogs out?"

    @pytest.mark.parametrize(
        "stop_words_to_add_back, expected",
        [
            ([""], ["let", "dogs"]),
            (["who"], ["Who", "let", "dogs"]),
            (["who", "the"], ["Who", "let", "the", "dogs"]),
            (["the", "out"], ["let", "the", "dogs", "out"]),
        ],
    )
    def test_reincluded_stop_word_for_word_embedding(
        self, sentence, stop_words_to_add_back, expected
    ):
        """
        Testing if adding back stop words function works
        """
        assert (
            preprocess_text_for_word_embedding(sentence, {}, 0, stop_words_to_add_back)
            == expected
        )

    def test_entity_recognition_simple(self):
        """
        Test parsing out of entities using simple dictionary
        """
        simple_dict = {("african", "union"): "African_Union"}

        various_cases = [
            "african union",
            "African union",
            "african Union",
            "AFRICAN UNION",
            "aFrican union",
            "aFRICAN UNion",
        ]

        for msg in various_cases:
            assert preprocess_text_for_word_embedding(msg, simple_dict, 0) == [
                "African_Union"
            ]

    def test_entity_recognition_in_sentence(self):
        """
        Test parsing out of entities from sentence, using simple dictionary
        """
        simple_dict = {("african", "union"): "African_Union"}

        desired_results = {
            "african union leadership structure": [
                "African_Union",
                "leadership",
                "structure",
            ],
            "is the african union based in Addis": [
                "African_Union",
                "based",
                "Addis",
            ],
            "how many countries are in the African UNION": [
                "many",
                "countries",
                "African_Union",
            ],
            "african union size": ["African_Union", "size"],
            "history AfricAn unIon": ["history", "African_Union"],
        }

        for msg in desired_results:
            assert (
                preprocess_text_for_word_embedding(msg, simple_dict, 0)
                == desired_results[msg]
            )

    def test_url_parsing_simple_for_word_embedding(self):
        desired_results = {
            "https://www.nytimes.com/video/multimedia/100000003081122/hot"
            "-commodity-"
            "the-data-scientist.html": [
                "hot",
                "commodity",
                "data",
                "scientist",
            ],
            "https://www.nytimes.com/2014/08/18/technology/for-big-data"
            "-scientists-"
            "hurdle-to-insights-is-janitor-work.html?searchResultPosition=3": [
                "big",
                "data",
                "scientists",
                "hurdle",
                "insights",
                "janitor",
                "work",
            ],
            "https://www.nytimes.com/2021/03/12/world/africa/senegal-female-"
            "empowerment-diouf-fishing.html?referringSource=articleShare": [
                "senegal",
                "female",
                "empowerment",
                "diouf",
                "fishing",
            ],
            "https://arxiv.org/abs/1412.6980": [],
            "https://www.the-scientist.com/news-opinion/lancet-retracts-"
            "surgispheres-study-on-hydroxychloroquine-67613": [
                "lancet",
                "retracts",
                "surgispheres",
                "study",
                "hydroxychloroquine",
                "67613",
            ],
        }

        for msg in desired_results:
            assert (
                preprocess_text_for_word_embedding(msg, {}, 4) == desired_results[msg]
            )

    def test_url_parsing_in_sentence_for_word_embedding(self):
        desired_results = {
            "i found this cool article "
            "https://www.nytimes.com/video/multimedia/"
            "100000003081122/hot-commodity-the-data-scientist.html": [
                "found",
                "cool",
                "article",
                "hot",
                "commodity",
                "data",
                "scientist",
            ],
            "https://www.nytimes.com/2014/08/18/technology/for-big-data"
            "-scientists-"
            "hurdle-to-insights-is-janitor-work.html?searchResultPosition=3 "
            "says that "
            "data wrangling is so boring!!": [
                "big",
                "data",
                "scientists",
                "hurdle",
                "insights",
                "janitor",
                "work",
                "says",
                "data",
                "wrangling",
                "boring",
            ],
            "women in senegal https://www.nytimes.com/2021/03/12/world/africa"
            "/senegal-"
            "female-empowerment-diouf-fishing.html?referringSource"
            "=articleShare "
            "empowerment": [
                "women",
                "senegal",
                "senegal",
                "female",
                "empowerment",
                "diouf",
                "fishing",
                "empowerment",
            ],
            "we should use Adam https://arxiv.org/abs/1412.6980": [
                "use",
                "Adam",
            ],
            "if you've ever been to a LMIC "
            "https://www.the-scientist.com/news-opinion/"
            "lancet-retracts-surgispheres-study-on-hydroxychloroquine-67613 "
            "was "
            "completely unsurprising": [
                "ever",
                "LMIC",
                "lancet",
                "retracts",
                "surgispheres",
                "study",
                "hydroxychloroquine",
                "67613",
                "completely",
                "unsurprising",
            ],
        }

        for msg in desired_results:
            assert (
                preprocess_text_for_word_embedding(msg, {}, 4) == desired_results[msg]
            )


class TestPreprocessingForKeywordRule:
    @pytest.fixture
    def vanilla_huns(self):
        return Hunspell()

    @pytest.fixture(scope="class")
    def porter_stemmer(self):
        return PorterStemmer()

    @pytest.fixture
    def porter_stem_func(self, porter_stemmer):
        return porter_stemmer.stem

    @pytest.fixture
    def identity_stem_func(self):
        return lambda x: x

    def test_punctuation_tokenization_stopwords(self, porter_stem_func, vanilla_huns):
        """
        Test removal of punctuation, tokenization, and removal of stopwords
        """
        desired_results = {
            "It was the BEST of times // it was the wORST of Times // it was "
            "the age of wIsDoM // it was the age of foolishness!!": [
                "best",
                "time",
                "worst",
                "time",
                "age",
                "wisdom",
                "age",
                "foolish",
            ],
            "'Shoot all the bluejays you want---if you can hit 'em, "
            "but remember it's "
            "a sin to kill_a_mockingbird.'": [
                "shoot",
                "blue",
                "jay",
                "want",
                "hit",
                "em",
                "rememb",
                "sin",
                "kill",
                "mockingbird",
            ],
            "Nobody, of the hundreds of people that had visited the Fair, "
            "knew that a "
            "grey spider had played the most important part of all.": [
                "nobodi",
                "hundr",
                "peopl",
                "visit",
                "fair",
                "knew",
                "grey",
                "spider",
                "play",
                "import",
                "part",
            ],
            "So it goes...": ["goe"],
        }

        for msg in desired_results:
            assert (
                preprocess_text_for_keyword_rule(
                    msg,
                    0,
                    stem_func=porter_stem_func,
                    spell_checker=vanilla_huns,
                    ngram_min=1,
                    ngram_max=1,
                )
                == desired_results[msg]
            )

    @pytest.fixture
    def sentence(self):
        return "Who let the dogs out?"

    @pytest.mark.parametrize(
        "stop_words_to_add_back, expected",
        [
            ([""], ["let", "dog"]),
            (["who"], ["who", "let", "dog"]),
            (["who", "the"], ["who", "let", "the", "dog"]),
            (["the", "out"], ["let", "the", "dog", "out"]),
        ],
    )
    def test_reincluded_stop_word_for_keyword_rule(
        self, sentence, porter_stem_func, vanilla_huns, stop_words_to_add_back, expected
    ):
        """
        Testing if adding back stop words function works
        """
        assert (
            preprocess_text_for_keyword_rule(
                sentence,
                0,
                stem_func=porter_stem_func,
                spell_checker=vanilla_huns,
                reincluded_stop_words=stop_words_to_add_back,
                ngram_min=1,
                ngram_max=1,
            )
            == expected
        )

    @pytest.mark.parametrize(
        "message, expected",
        [
            (
                "https://www.nytimes.com/video/multimedia/100000003081122/hot"
                "-commodity-the-data-scientist.html",
                [
                    "hot",
                    "commod",
                    "data",
                    "scientist",
                ],
            ),
            (
                "https://www.nytimes.com/2014/08/18/technology/for-big-data"
                "-scientists-"
                "hurdle-to-insights-is-janitor-work.html?searchResultPosition=3",
                [
                    "big",
                    "data",
                    "scientist",
                    "hurdl",
                    "insight",
                    "janitor",
                    "work",
                ],
            ),
            (
                "https://www.nytimes.com/2021/03/12/world/africa/senegal-female-"
                "empowerment-diouf-fishing.html?referringSource=articleShare",
                [
                    "seneg",
                    "femal",
                    "empower",
                    "odiou",
                    "fish",
                ],
            ),
            ("https://arxiv.org/abs/1412.6980", []),
            (
                "https://www.the-scientist.com/news-opinion/lancet-retracts-"
                "surgispheres-study-on-hydroxychloroquine-67613",
                [
                    "lancet",
                    "retract",
                    "hemispher",
                    "studi",
                    "hydroxi",
                    "chloroquin",
                    "67613",
                ],
            ),
            (
                "https://www.givingwhatwecan.org/get-involved/birthday" "-fundraisers/",
                [],
            ),
        ],
    )
    def test_url_parsing_simple_for_keyword_rule(
        self, porter_stem_func, vanilla_huns, message, expected
    ):
        assert (
            preprocess_text_for_keyword_rule(
                message,
                4,
                stem_func=porter_stem_func,
                spell_checker=vanilla_huns,
                reincluded_stop_words=None,
                ngram_min=1,
                ngram_max=1,
            )
            == expected
        )

    @pytest.mark.parametrize(
        "message, expected",
        [
            (
                "i found this cool article https://www.nytimes.com/video/multimedia/"
                "100000003081122/hot-commodity-the-data-scientist.html",
                [
                    "found",
                    "cool",
                    "articl",
                    "hot",
                    "commod",
                    "data",
                    "scientist",
                ],
            ),
            (
                "https://www.nytimes.com/2014/08/18/technology/for-big-data-scientists-"
                "hurdle-to-insights-is-janitor-work.html?searchResultPosition=3 says "
                "that data wrangling is so boring!!",
                [
                    "big",
                    "data",
                    "scientist",
                    "hurdl",
                    "insight",
                    "janitor",
                    "work",
                    "say",
                    "data",
                    "wrangl",
                    "bore",
                ],
            ),
            (
                "women in senegal https://www.nytimes.com/2021/03/12/world/africa"
                "/senegal-"
                "female-empowerment-diouf-fishing.html?referringSource=articleShare "
                "empowerment",
                [
                    "women",
                    "seneg",
                    "seneg",
                    "femal",
                    "empower",
                    "odiou",
                    "fish",
                    "empower",
                ],
            ),
            (
                "we should use Adam https://arxiv.org/abs/1412.6980",
                [
                    "use",
                    "adam",
                ],
            ),
            (
                "if you've ever been to a LMIC "
                "https://www.the-scientist.com/news-opinion/"
                "lancet-retracts-surgispheres-study-on-hydroxychloroquine-67613 was "
                "completely unsurprising",
                [
                    "ever",
                    "mimic",
                    "lancet",
                    "retract",
                    "hemispher",
                    "studi",
                    "hydroxi",
                    "chloroquin",
                    "67613",
                    "complet",
                    "unsurpris",
                ],
            ),
            (
                "Have you seen this: https://www.givingwhatwecan.org/get-involved"
                "/birthday"
                "-fundraisers/",
                [
                    "seen",
                ],
            ),
        ],
    )
    def test_url_parsing_in_sentence_for_keyword_rule(
        self, porter_stem_func, vanilla_huns, message, expected
    ):
        assert (
            preprocess_text_for_keyword_rule(
                message,
                4,
                stem_func=porter_stem_func,
                spell_checker=vanilla_huns,
                reincluded_stop_words=None,
                ngram_min=1,
                ngram_max=1,
            )
            == expected
        )

    def test_custom_spell_correct_map(self, porter_stem_func):
        message = "Help I think my baby has jondis"
        spell_correct_map = {"jondis": "jaundice", "pewking": "puking"}
        huns = CustomHunspell(custom_spell_correct_map=spell_correct_map)

        assert preprocess_text_for_keyword_rule(
            message, 0, stem_func=porter_stem_func, spell_checker=huns, ngram_max=1
        ) == ["help", "think", "babi", "jaundic"]

    def test_custom_spell_check_list(self, identity_stem_func, porter_stem_func):
        message = "Stop texting me pls"
        spell_check_list = ["texting", "pls"]
        huns = CustomHunspell(custom_spell_check_list=spell_check_list)

        assert preprocess_text_for_keyword_rule(
            message, 0, stem_func=identity_stem_func, spell_checker=huns, ngram_max=1
        ) == ["stop", "texting", "pls"]

        assert preprocess_text_for_keyword_rule(
            message, 0, stem_func=porter_stem_func, spell_checker=huns, ngram_max=1
        ) == ["stop", "text", "pl"]

    @pytest.mark.parametrize(
        "message, expected",
        [
            (
                "I had pian in virginia for the last 2 wks help",
                ["pain", "virginia", "last", "2", "wk", "help"],
            ),
            ("My chils is burping alot what should I DO?", ["child", "burp", "alto"]),
        ],
    )
    def test_priority_words(self, porter_stem_func, message, expected):
        huns = CustomHunspell(priority_words=["pain", "child"])

        assert (
            preprocess_text_for_keyword_rule(
                message, 0, stem_func=porter_stem_func, spell_checker=huns, ngram_max=1
            )
            == expected
        )

    @pytest.mark.parametrize(
        "message, expected",
        [
            (
                "I had pian in virginia for the last 2 wks help",
                [
                    "pain",
                    "vagina",
                    "last",
                    "2",
                    "week",
                    "help",
                    "pain_vagina",
                    "vagina_last",
                    "last_2",
                    "2_week",
                    "week_help",
                ],
            ),
            (
                "My chils is burping alot what should I DO?",
                [
                    "child",
                    "burp",
                    "lot",
                    "should",
                    "child_burp",
                    "burp_lot",
                    "lot_should",
                ],
            ),
            (
                "if you've ever been to a LMIC https://www.the-scientist.com/news-opinion/"
                "lancet-retracts-surgispheres-study-on-hydroxychloroquine-67613 was "
                "completely unsurprising",
                [
                    "ever",
                    "lmic",
                    "lancet",
                    "retract",
                    "hemispher",
                    "studi",
                    "hydroxi",
                    "chloroquin",
                    "67613",
                    "complet",
                    "unsurpris",
                    "ever_lmic",
                    "lmic_lancet",
                    "lancet_retract",
                    "retract_hemispher",
                    "hemispher_studi",
                    "studi_hydroxi",
                    "hydroxi_chloroquin",
                    "chloroquin_67613",
                    "67613_complet",
                    "complet_unsurpris",
                ],
            ),
            (
                "Have you seen this 🌊: "
                "https://www.givingwhatwecan.org/get-involved/birthday-fundraisers/",
                [
                    "seen",
                ],
            ),
            (
                "Help I think my baby has jondis",
                [
                    "help",
                    "think",
                    "babi",
                    "jaundic",
                    "help_think",
                    "think_babi",
                    "babi_jaundic",
                ],
            ),
            (
                "Stop texting me pls 😒",
                ["stop", "text", "pleas", "stop_text", "text_pleas"],
            ),
        ],
    )
    def test_complex(self, porter_stem_func, message, expected):
        huns = CustomHunspell(
            custom_spell_check_list=["texting", "texter", "lmic"],
            custom_spell_correct_map={
                "virginia": "vagina",
                "jondis": "jaundice",
                "wks": "weeks",
                "alot": "a lot",
                "pls": "please",
            },
            priority_words=[
                "pain",
                "child",
            ],
        )

        assert (
            preprocess_text_for_keyword_rule(
                message,
                n_min_dashed_words_url=4,
                stem_func=porter_stem_func,
                spell_checker=huns,
                reincluded_stop_words=["should"],
                ngram_min=1,
                ngram_max=2,
            )
            == expected
        )

    @pytest.mark.parametrize(
        "message",
        [
            "https://www.givingwhatwecan.org/get-involved/birthday-fundraisers/",
            "🆗",
            "",
            "¥€$$$ !!",
        ],
    )
    def test_empty(self, message, identity_stem_func, vanilla_huns):
        assert (
            preprocess_text_for_keyword_rule(
                message,
                n_min_dashed_words_url=3,
                stem_func=identity_stem_func,
                spell_checker=vanilla_huns,
                ngram_min=1,
                ngram_max=2,
            )
            == []
        )
