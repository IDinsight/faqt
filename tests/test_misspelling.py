# Tests for handling misspellings in model (getting vectors, calculating CS)

# Tests are adapted from and very similar to those in the TestModel class
# (/tests/tests_model.py)

import numpy as np
import pytest
from faqt.model.embeddings import model_search, model_search_word

pytestmark = pytest.mark.slow


def test_model_search_word(w2v_model, hunspell):
    """
    Check that our model_search_word function (which searches for given case,
    title case, then lowercase) finds vector as expected

    Note that lowercase and title case vectors in model are different for nearly
    all words - e.g., w2v_model["movie"] != w2v_model["Movie"] - and they often
    even don't have high cosine similarity!

    Use np.allclose here to check equality since dealing with floats
    """
    # Should correct spelling to movie; movee not in w2v_model
    assert np.allclose(
        model_search_word(
            "movee",
            model=w2v_model,
            glossary={},
            hunspell=hunspell,
            tags_guiding_typos_wv=model_search(["television"], w2v_model, {}),
        ),
        w2v_model["movie"],
    )

    # Should correct spelling to movie; movee not in w2v_model
    assert np.allclose(
        model_search_word(
            "movee",
            model=w2v_model,
            glossary={},
            hunspell=hunspell,
            tags_guiding_typos_wv=model_search(
                ["television", "banana", "violin"], w2v_model, {}
            ),
        ),
        w2v_model["movie"],
    )

    # Should correct spelling to prison;
    # even though prism is closer per Hunspell
    assert np.allclose(
        model_search_word(
            "prisn",
            model=w2v_model,
            glossary={},
            hunspell=hunspell,
            tags_guiding_typos_wv=model_search(["jail"], w2v_model, {}),
        ),
        w2v_model["prison"],
    )

    # Should correct spelling to USA,
    # including changing the case
    assert np.allclose(
        model_search_word(
            "usaa",
            model=w2v_model,
            glossary={},
            hunspell=hunspell,
            tags_guiding_typos_wv=model_search(["America"], w2v_model, {}),
        ),
        w2v_model["USA"],
    )


def test_model_search_word_negative(w2v_model, hunspell):
    """
    Since all of these terms are in model, model_search_word shouldn't correct spelling
    """
    # Since move is in model, shouldn't correct spelling
    assert not np.allclose(
        model_search_word(
            "move",
            model=w2v_model,
            glossary={},
            hunspell=hunspell,
            tags_guiding_typos_wv=model_search(["television"], w2v_model, {}),
        ),
        w2v_model["movie"],
    )

    # Since move is in model, shouldn't correct spelling
    assert not np.allclose(
        model_search_word(
            "move",
            model=w2v_model,
            glossary={},
            hunspell=hunspell,
            tags_guiding_typos_wv=model_search(
                ["television", "banana", "violin"], w2v_model, {}
            ),
        ),
        w2v_model["movie"],
    )


def test_model_search_word_no_alternatives(w2v_model, hunspell):
    """
    Check that our model_search_word function (which searches for given case,
    title case, then lowercase) doesn't find vector, since no alternative spellings
    """
    # No Hunspell alternatives
    assert (
        model_search_word(
            "thisisnotarealword",
            model=w2v_model,
            glossary={},
            hunspell=hunspell,
            tags_guiding_typos_wv=model_search(
                ["television", "banana", "violin"], w2v_model, {}
            ),
        )
        is None
    )

    # No Hunspell alternatives
    assert (
        model_search_word(
            "stillNotARealWord",
            model=w2v_model,
            glossary={},
            hunspell=hunspell,
            tags_guiding_typos_wv=model_search(
                ["television", "banana", "violin"], w2v_model, {}
            ),
        )
        is None
    )


def test_model_search_word_glossary_single_component(w2v_model, hunspell):
    """
    Check that our model_search_word function (which searches first in glossary,
    then given case, title case, and lowercase) finds vector as expected

    Note that lowercase and title case vectors in model are different for nearly
    all words - e.g., w2v_model["movie"] != w2v_model["Movie"] - and they often
    even don't have high cosine similarity!

    Use np.allclose here to check equality since dealing with floats
    """
    custom_glossary = {
        "trial": {"error": 1},
        "banana": {"big": 0.5, "fruit": 0.5},
        "dumb-tag": {"smart": 1},
    }

    # Our misspelling algorithm should correct tryal -> trial,
    # with the clinical tag, but still default first to custom_glossary
    assert not np.allclose(
        model_search_word(
            "tryal",
            model=w2v_model,
            glossary=custom_glossary,
            hunspell=hunspell,
            tags_guiding_typos_wv=model_search(["clinical"], w2v_model, {}),
        ),
        w2v_model["trial"],
    )

    # Our misspelling algorithm should correct tryal -> trial,
    # with the clinical tag, but still default first to custom_glossary
    assert np.allclose(
        model_search_word(
            "tryal",
            model=w2v_model,
            glossary=custom_glossary,
            hunspell=hunspell,
            tags_guiding_typos_wv=model_search(["clinical"], w2v_model, {}),
        ),
        w2v_model["error"],
    )

    # Our misspelling algorithm should correct banan -> banana,
    # with the fruit tag, but still default first to custom_glossary
    assert not np.allclose(
        model_search_word(
            "banan",
            model=w2v_model,
            glossary=custom_glossary,
            hunspell=hunspell,
            tags_guiding_typos_wv=model_search(["fruit"], w2v_model, {}),
        ),
        w2v_model["banana"],
    )


def test_model_search_word_glossary_multiple_components(w2v_model, hunspell):
    custom_glossary = {
        "trial": {"error": 1},
        "banana": {"big": 0.5, "fruit": 0.5},
        "dumb-tag": {"smart": 1},
    }

    banana_components = 0.5 * w2v_model["big"] + 0.5 * w2v_model["fruit"]

    # Our misspelling algorithm should correct banan -> banana,
    # with the fruit tag, but still default first to custom_glossary
    assert np.allclose(
        model_search_word(
            "abnana",
            model=w2v_model,
            glossary=custom_glossary,
            hunspell=hunspell,
            tags_guiding_typos_wv=model_search(["fruit"], w2v_model, {}),
        ),
        banana_components / np.sqrt(np.dot(banana_components, banana_components)),
    )

    # Spelling corrections for "dumbtag" are "dumb tag", "dumb-tag", and
    # "thumbtack." Even though "dumb-tag" is a closer misspelling, and is
    # in the contextualization glossary, "thumbtack" is clearly the better
    # match to "pin."
    assert np.allclose(
        model_search_word(
            "dumbtag",
            model=w2v_model,
            glossary=custom_glossary,
            hunspell=hunspell,
            tags_guiding_typos_wv=model_search(["pin"], w2v_model, {}),
        ),
        w2v_model["thumbtack"],
    )
