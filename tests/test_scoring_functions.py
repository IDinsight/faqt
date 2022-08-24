import math

import numpy as np
import pytest
from faqt.model.faq_matching.keyed_vector_scoring import model_search, model_search_word
from faqt.scoring.single_tag_scoring import cs_nearest_k_percent_average

# Need to tune these dimensions if not using Google News pretrained model
LOW_CS = 0.2
HIGH_CS = 0.8


@pytest.mark.fast
def test_cs_basic():
    """
    Test cosine similarity calculation on easy paper-and-pencil vectors
    """
    ONE_DIV_ROOT2 = 1 / np.sqrt(2)

    assert np.isclose(
        cs_nearest_k_percent_average(
            [np.array([-ONE_DIV_ROOT2, -ONE_DIV_ROOT2])] * 3,
            target_wv=np.array([ONE_DIV_ROOT2, ONE_DIV_ROOT2]),
            k=100,
            floor=1,
        ),
        -1,
    )

    assert np.isclose(
        cs_nearest_k_percent_average(
            [np.array([ONE_DIV_ROOT2, -ONE_DIV_ROOT2])] * 3,
            target_wv=np.array([ONE_DIV_ROOT2, ONE_DIV_ROOT2]),
            k=100,
            floor=1,
        ),
        0,
    )

    assert np.isclose(
        cs_nearest_k_percent_average(
            [np.array([ONE_DIV_ROOT2, ONE_DIV_ROOT2])] * 3,
            target_wv=np.array([ONE_DIV_ROOT2, ONE_DIV_ROOT2]),
            k=100,
            floor=1,
        ),
        1,
    )


@pytest.mark.fast
def test_cs_with_k():
    """
    Test cosine similarity calculation on some paper-and-pencil vectors, where
    k < 100 (only selecting the closest vectors)
    """
    ONE_DIV_ROOT2 = 1 / np.sqrt(2)

    assert np.isclose(
        cs_nearest_k_percent_average(
            [np.array([ONE_DIV_ROOT2, ONE_DIV_ROOT2])]
            + [np.array([-ONE_DIV_ROOT2, -ONE_DIV_ROOT2])] * 9,
            target_wv=np.array([ONE_DIV_ROOT2, ONE_DIV_ROOT2]),
            k=10,
            floor=1,
        ),
        1,
    )


@pytest.mark.fast
def test_cs_averaging():
    """
    Test cosine similarity averaging on some paper-and-pencil vectors
    """
    ONE_DIV_ROOT2 = 1 / np.sqrt(2)

    # Should take average to be [0, 0], which has 0 CS
    assert np.isclose(
        cs_nearest_k_percent_average(
            [np.array([ONE_DIV_ROOT2, ONE_DIV_ROOT2])]
            + [np.array([-ONE_DIV_ROOT2, -ONE_DIV_ROOT2])] * 9,
            target_wv=np.array([ONE_DIV_ROOT2, ONE_DIV_ROOT2]),
            k=10,
            floor=2,
        ),
        0,
    )

    # Should take average to be [0, 1], which has 1/sqrt(2) CS with [1, 1]
    assert np.isclose(
        cs_nearest_k_percent_average(
            [np.array([0, ONE_DIV_ROOT2])] * 10,
            target_wv=np.array([ONE_DIV_ROOT2, ONE_DIV_ROOT2]),
            k=10,
            floor=1,
        ),
        1 / math.sqrt(2),
    )

    # Should take average to be [0, 1], which has 1/sqrt(2) CS with [1, 1]
    assert np.isclose(
        cs_nearest_k_percent_average(
            [np.array([ONE_DIV_ROOT2, ONE_DIV_ROOT2])]
            + [np.array([-ONE_DIV_ROOT2, ONE_DIV_ROOT2])] * 9,
            target_wv=np.array([ONE_DIV_ROOT2, ONE_DIV_ROOT2]),
            k=10,
            floor=2,
        ),
        1 / math.sqrt(2),
    )

    # Should take average to be [0, 1], which has 1/sqrt(2) CS with [1, 1]
    assert np.isclose(
        cs_nearest_k_percent_average(
            [np.array([ONE_DIV_ROOT2, ONE_DIV_ROOT2])]
            + [np.array([-ONE_DIV_ROOT2, ONE_DIV_ROOT2])] * 9,
            target_wv=np.array([ONE_DIV_ROOT2, ONE_DIV_ROOT2]),
            k=20,
            floor=1,
        ),
        1 / math.sqrt(2),
    )


@pytest.mark.slow
def test_cs_model_basic(w2v_model):
    """
    Test cosine similarity calculation using some simple sentences + model
    """
    assert np.isclose(
        cs_nearest_k_percent_average(
            list_of_wvs=model_search(["I", "saw", "a", "chicken"], w2v_model, {}),
            target_wv=model_search_word("chicken", model=w2v_model, glossary={}),
            k=10,
            floor=1,
        ),
        1,
    )

    assert (
        cs_nearest_k_percent_average(
            list_of_wvs=model_search(["I", "saw", "a", "movie"], w2v_model, {}),
            target_wv=model_search_word("film", model=w2v_model, glossary={}),
            k=10,
            floor=1,
        )
        > HIGH_CS
    )

    assert (
        cs_nearest_k_percent_average(
            list_of_wvs=model_search(["I", "saw", "a", "movie"], w2v_model, {}),
            target_wv=model_search_word("pig", model=w2v_model, glossary={}),
            k=10,
            floor=1,
        )
        < LOW_CS
    )


@pytest.mark.slow
def test_cs_model_long_sentences(w2v_model):
    """
    Test cosine similarity calculation using some simple sentences + model
    """
    assert (
        cs_nearest_k_percent_average(
            list_of_wvs=model_search(["I", "saw", "a", "movie"] * 100, w2v_model, {}),
            target_wv=model_search_word("film", model=w2v_model, glossary={}),
            k=25,
            floor=1,
        )
        > HIGH_CS
    )

    assert (
        cs_nearest_k_percent_average(
            list_of_wvs=model_search(["I", "saw", "a", "movie"] * 100, w2v_model, {}),
            target_wv=model_search_word("pig", model=w2v_model, glossary={}),
            k=25,
            floor=1,
        )
        < LOW_CS
    )


@pytest.mark.slow
def test_cs_model_long_sentences_high_floor(w2v_model):
    """
    Test cosine similarity calculation using some simple sentences + model
    """
    assert (
        cs_nearest_k_percent_average(
            list_of_wvs=model_search(["I", "saw", "a", "movie"] * 100, w2v_model, {}),
            target_wv=model_search_word("film", model=w2v_model, glossary={}),
            k=25,
            floor=75,
        )
        > HIGH_CS
    )

    assert (
        cs_nearest_k_percent_average(
            list_of_wvs=model_search(["I", "saw", "a", "movie"] * 100, w2v_model, {}),
            target_wv=model_search_word("pig", model=w2v_model, glossary={}),
            k=25,
            floor=75,
        )
        < LOW_CS
    )

    assert (
        cs_nearest_k_percent_average(
            list_of_wvs=model_search(
                ["I", "saw", "a", "movie", "it", "was", "really", "good"] * 100,
                w2v_model,
                {},
            ),
            target_wv=model_search_word("film", model=w2v_model, glossary={}),
            k=10,
            floor=75,
        )
        > HIGH_CS
    )

    assert (
        cs_nearest_k_percent_average(
            list_of_wvs=model_search(
                ["I", "saw", "a", "movie", "it", "was", "really", "good"] * 100,
                w2v_model,
                {},
            ),
            target_wv=model_search_word("movie", model=w2v_model, glossary={}),
            k=50,
            floor=75,
        )
        < HIGH_CS
    )


@pytest.mark.slow
def test_cs_model_ordinal(w2v_model):
    """
    Test cosine similarity calculation using ordinal relations
    """
    cs_movie_film = cs_nearest_k_percent_average(
        list_of_wvs=model_search(["I", "saw", "a", "film"], w2v_model, {}),
        target_wv=model_search_word("movie", model=w2v_model, glossary={}),
        k=10,
        floor=1,
    )

    cs_camera_film = cs_nearest_k_percent_average(
        list_of_wvs=model_search(["I", "saw", "a", "film"], w2v_model, {}),
        target_wv=model_search_word("camera", model=w2v_model, glossary={}),
        k=10,
        floor=1,
    )

    cs_banana_film = cs_nearest_k_percent_average(
        list_of_wvs=model_search(["I", "saw", "a", "film"], w2v_model, {}),
        target_wv=model_search_word("banana", model=w2v_model, glossary={}),
        k=10,
        floor=1,
    )

    assert cs_movie_film > cs_camera_film
    assert cs_camera_film > cs_banana_film
