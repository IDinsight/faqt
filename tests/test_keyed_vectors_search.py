import math

import numpy as np
import pytest
from faqt.model.faq_matching.keyed_vectors_scoring import (
    model_search,
    model_search_word,
)

LOW_CS = 0.2
HIGH_CS = 0.8


@pytest.mark.slow
def test_pretrained_existence(w2v_model):
    """
    Check words are in/not in word_embedding_model, dimension correct
    """
    assert "horse" in w2v_model
    assert "notarealword" not in w2v_model


@pytest.mark.slow
def test_pretrained_similarities(w2v_model):
    """
    Check similarities in word_embedding_model
    """

    def cos_sim(x, y):
        return sum(x * y) / (math.sqrt(sum(x * x) * math.sqrt(sum(y * y))))

    vec_movie = w2v_model["movie"]
    vec_film = w2v_model["film"]
    vec_orange = w2v_model["orange"]

    assert cos_sim(vec_movie, vec_film) > HIGH_CS
    assert cos_sim(vec_movie, vec_orange) < LOW_CS


@pytest.mark.slow
def test_model_search_word(w2v_model):
    """
    Check that our model_search_word function (which searches for given case,
    lowercase, title case) finds vector as expected

    Note that lowercase and title case vectors in word_embedding_model are different for nearly
    all words - e.g., word_embedding_model["movie"] != word_embedding_model["Movie"] - and they often
    even don't have high cosine similarity!

    Use np.allclose here to check equality since dealing with floats
    """
    assert np.allclose(model_search_word("mOvIE", w2v_model, {}), w2v_model["movie"])
    assert not np.allclose(model_search_word("film", w2v_model, {}), w2v_model["movie"])

    assert model_search_word("notarealword", w2v_model, {}) is None
    assert model_search_word("stillNotARealWord", w2v_model, {}) is None


@pytest.mark.slow
def test_model_search_word_case(w2v_model):
    """
    Check that our model_search_word function (which searches for given case,
    lowercase, title case) finds vector as expected

    Note that lowercase and title case vectors in word_embedding_model are different for nearly
    all words - e.g., word_embedding_model["movie"] != word_embedding_model["Movie"] - and they often
    even don't have high cosine similarity!

    Use np.allclose here to check equality since dealing with floats
    """
    # movie (lowercase) is in word_embedding_model search word, so should be different
    assert not np.allclose(
        model_search_word("movie", w2v_model, {}), w2v_model["Movie"]
    )
    # muhammadu (lowercase) is not in word_embedding_model, title case is, so should return
    # title case
    assert np.allclose(
        model_search_word("muHAMMADu", w2v_model, {}), w2v_model["Muhammadu"]
    )

    # only lowercase is in the word_embedding_model
    assert np.allclose(
        model_search_word("CoNoCiMiEnTo", w2v_model, {}),
        w2v_model["conocimiento"],
    )


@pytest.mark.slow
def test_model_search_word_glossary_single_component(w2v_model):
    """
    Check that our model_search_word function (which searches first in glossary)
    finds vector as expected

    Use np.allclose here to check equality since dealing with floats
    """
    custom_glossary = {
        "trial": {"error": 1},
        "banana": {"big": 0.5, "fruit": 0.5},
        "notarealword": {"fake": 0.2, "word": 0.8},
    }

    assert not np.allclose(
        model_search_word("trial", w2v_model, custom_glossary),
        w2v_model["trial"],
    )

    assert np.allclose(
        model_search_word("trial", w2v_model, custom_glossary),
        w2v_model["error"],
    )

    assert not np.allclose(
        model_search_word("banana", w2v_model, custom_glossary),
        w2v_model["banana"],
    )


@pytest.mark.slow
def test_model_search_word_glossary_multiple_components(w2v_model):
    custom_glossary = {
        "trial": {"error": 1},
        "banana": {"big": 0.5, "fruit": 0.5},
        "notarealword": {"fake": 0.2, "word": 0.8},
    }

    banana_components = 0.5 * w2v_model["big"] + 0.5 * w2v_model["fruit"]

    assert np.allclose(
        model_search_word("banana", w2v_model, custom_glossary),
        banana_components / np.sqrt(np.dot(banana_components, banana_components)),
    )

    assert model_search_word("notarealword", w2v_model, {}) is None
    assert model_search_word("notarealword", w2v_model, custom_glossary) is not None

    fake_word_05_05 = 0.5 * w2v_model["fake"] + 0.5 * w2v_model["word"]
    assert not np.allclose(
        model_search_word("notarealword", w2v_model, custom_glossary),
        fake_word_05_05 / np.sqrt(np.dot(fake_word_05_05, fake_word_05_05)),
    )

    fake_word_02_08 = 0.2 * w2v_model["fake"] + 0.8 * w2v_model["word"]
    assert np.allclose(
        model_search_word("notarealword", w2v_model, custom_glossary),
        fake_word_02_08 / np.sqrt(np.dot(fake_word_02_08, fake_word_02_08)),
    )


@pytest.mark.slow
def test_model_search_sentence_length(w2v_model):
    """
    Check that our model_search function (which searches first in glossary,
    then given case, title case, and lowercase) finds vectors as expected,
    for multiple words in sentence
    """
    assert len(model_search(["movie", "star", "star"], w2v_model, {})) == 3
    assert len(model_search(["movie"] + ["star"] * 107, w2v_model, {})) == 108


@pytest.mark.slow
def test_model_search_sentence_vectors(w2v_model):
    """
    Check that our model_search function (which searches first in glossary,
    then given case, title case, and lowercase) finds correct vectors,
    for multiple words in sentence

    Use np.allclose here to check equality since dealing with floats
    """
    assert np.allclose(
        model_search(["movie", "star"], w2v_model, {}),
        [w2v_model["movie"], w2v_model["star"]],
    )

    assert np.allclose(
        model_search(["movie", "star", "studio"], w2v_model, {}),
        [w2v_model["movie"], w2v_model["star"], w2v_model["studio"]],
    )

    assert np.allclose(
        model_search(["experimental", "trial"], w2v_model, {}),
        [w2v_model["experimental"], w2v_model["trial"]],
    )

    assert not np.allclose(
        model_search(["experimental", "trial"], w2v_model, {}),
        [w2v_model["experimental"], w2v_model["error"]],
    )


@pytest.mark.slow
def test_model_search_sentence_case_correction(w2v_model):
    """
    Check that our model_search function (which searches first in glossary,
    then given case, title case, and lowercase) finds vectors as expected,
    for multiple words in sentence

    Note that lowercase and title case vectors in word_embedding_model are different for nearly
    all words - e.g., word_embedding_model["movie"] != word_embedding_model["Movie"] - and they often even
    don't have high cosine similarity!

    Use np.allclose here to check equality since dealing with floats
    """
    assert np.allclose(
        model_search(["knowledge", "cOnOcImIenTO"], w2v_model, {}),
        [w2v_model["knowledge"], w2v_model["conocimiento"]],
    )

    assert np.allclose(
        model_search(["muhaMMadu", "president"], w2v_model, {}),
        [w2v_model["Muhammadu"], w2v_model["president"]],
    )


@pytest.mark.slow
def test_model_search_sentence_glossary(w2v_model):
    """
    Check that our model_search function (which searches first in glossary)
    finds vectors as expected for multiple words in sentence

    Use np.allclose here to check equality since dealing with floats
    """
    custom_glossary = {
        "trial": {"error": 1},
        "banana": {"big": 0.5, "fruit": 0.5},
        "notarealword": {"fake": 0.2, "word": 0.8},
    }

    assert np.allclose(
        model_search(["experimental", "tRiAl"], w2v_model, custom_glossary),
        [w2v_model["experimental"], w2v_model["error"]],
    )

    assert not np.allclose(
        model_search(["experimental", "tRiAl"], w2v_model, custom_glossary),
        [w2v_model["experimental"], w2v_model["trial"]],
    )
