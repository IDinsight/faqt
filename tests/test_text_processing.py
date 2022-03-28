import pytest
from faqt.preprocessing import preprocess_text

pytestmark = pytest.mark.fast


class TestPunctuation:
    def test_punctuation_tokenization_stopwords(self):
        """
        Test removal of punctuation, tokenization, and removal of stopwords
        """
        desired_results = {
            "It was the BEST of times // it was the wORST of Times // it was the age "
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
            "'Shoot all the bluejays you want---if you can hit 'em, but remember it's "
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
            "Nobody, of the hundreds of people that had visited the Fair, knew that a "
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
            assert preprocess_text(msg, {}, 0) == desired_results[msg]


class TestIncludingStopWords:
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
    def test_reincluded_stop_word(self, sentence, stop_words_to_add_back, expected):
        """
        Testing if adding back stop words function works
        """
        assert preprocess_text(sentence, {}, 0, stop_words_to_add_back) == expected


class TestEntityRecognition:
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
            assert preprocess_text(msg, simple_dict, 0) == ["African_Union"]

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
            assert preprocess_text(msg, simple_dict, 0) == desired_results[msg]


class TestUrlParsing:
    def test_url_parsing_simple(self):
        desired_results = {
            "https://www.nytimes.com/video/multimedia/100000003081122/hot-commodity-"
            "the-data-scientist.html": [
                "hot",
                "commodity",
                "data",
                "scientist",
            ],
            "https://www.nytimes.com/2014/08/18/technology/for-big-data-scientists-"
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
            assert preprocess_text(msg, {}, 4) == desired_results[msg]

    def test_url_parsing_in_sentence(self):
        desired_results = {
            "i found this cool article https://www.nytimes.com/video/multimedia/"
            "100000003081122/hot-commodity-the-data-scientist.html": [
                "found",
                "cool",
                "article",
                "hot",
                "commodity",
                "data",
                "scientist",
            ],
            "https://www.nytimes.com/2014/08/18/technology/for-big-data-scientists-"
            "hurdle-to-insights-is-janitor-work.html?searchResultPosition=3 says that "
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
            "women in senegal https://www.nytimes.com/2021/03/12/world/africa/senegal-"
            "female-empowerment-diouf-fishing.html?referringSource=articleShare "
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
            "if you've ever been to a LMIC https://www.the-scientist.com/news-opinion/"
            "lancet-retracts-surgispheres-study-on-hydroxychloroquine-67613 was "
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
            assert preprocess_text(msg, {}, 4) == desired_results[msg]
