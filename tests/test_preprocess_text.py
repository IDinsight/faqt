import pytest

from faqt.preprocessing.text import process_urls, remove_punctuation


class TestUrlParsing:
    @pytest.mark.parametrize(
        "msg, expected",
            [
                ("https://www.nytimes.com/video/multimedia/100000003081122/hot"
            "-commodity-"
            "the-data-scientist.html", "hot-commodity-the-data-scientist"),
            ("https://www.nytimes.com/2014/08/18/technology/for-big-data"
            "-scientists-"
            "hurdle-to-insights-is-janitor-work.html?searchResultPosition=3",
                "for-big-data-scientists-hurdle-to-insights-is-janitor-work"),
            ("https://www.nytimes.com/2021/03/12/world/africa/senegal-female-"
            "empowerment-diouf-fishing.html?referringSource=articleShare",
                "senegal-female-empowerment-diouf-fishing"),
            ("https://arxiv.org/abs/1412.6980", ""),
            ("https://www.the-scientist.com/news-opinion/lancet-retracts-"
            "surgispheres-study-on-hydroxychloroquine-67613",
                "lancet-retracts-surgispheres-study-on-hydroxychloroquine"
                "-67613"),
            ("https://www.givingwhatwecan.org/get-involved/birthday"
            "-fundraisers/", "")]
    )
    def test_url_parsing_simple(self, msg, expected):
        assert process_urls(msg, 4) == expected

    @pytest.mark.parametrize(
        "desired_results",
        {"i found this cool article https://www.nytimes.com/video/multimedia/"
            "100000003081122/hot-commodity-the-data-scientist.html":
                "found cool article hot-commodity-the-data-scientist",
            "https://www.nytimes.com/2014/08/18/technology/for-big-data-scientists-"
            "hurdle-to-insights-is-janitor-work.html?searchResultPosition=3 says that "
            "data wrangling is so boring!!":
                "for-big-data-scientists-hurdle-to-insights-is-janitor-work "
                "says data wrangling boring",
            "women in senegal https://www.nytimes.com/2021/03/12/world/africa/senegal-"
            "female-empowerment-diouf-fishing.html?referringSource=articleShare "
            "empowerment":
                "women in senegal senegal-female-empowerment-diouf-fishing "
                "empowerment",

            "we should use Adam https://arxiv.org/abs/1412.6980":
                "we should use Adam",

            "if you've ever been to a LMIC https://www.the-scientist.com/news-opinion/"
            "lancet-retracts-surgispheres-study-on-hydroxychloroquine-67613 was "
            "completely unsurprising":
                "if you've ever been to a LMIC "
                "lancet-retracts-surgispheres-study-on-hydroxychloroquine-67613 "
                "was completely unsurprising"}
    )
    def test_url_parsing_in_sentence(self, desired_results):
        for msg in desired_results:
            assert process_urls(msg, 4) == desired_results[msg]


class TestPunctuation:
    def test_punctuation_tokenization_stopwords(self):
        """
        Test removal of punctuation, tokenization, and removal of stopwords
        """
        desired_results = {
            "It was the BEST of times // it was the wORST of Times // it was the age "
            "of wIsDoM // it was the age of foolishness!!": 
                "It was the BEST times it was the wORST of Times it was the "
                "age wIsDoM it was the age foolishness",
            "'Shoot all the bluejays you want---if you can hit 'em, but remember it's "
            "a sin to kill_a_mockingbird.'": 
                "Shoot all the bluejays you want if you can hit em but "
                "remember it s a sin to kill a mockingbird",
            
            "Nobody, of the hundreds of people that had visited the Fair, knew that a "
            "grey spider had played the most important part of all.":
                "Nobody of the hundreds of people that had visited Fair knew "
                "that a grey spider had played the most important part of all",
            "So it goes...": "So it goes",
        }

        for msg in desired_results:
            assert remove_punctuation(msg) == desired_results[msg]
