import os
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors


def load_wv_pretrained_bin(folder, filename):
    """
    Load pretrained word2vec model from either local mount or S3
    based on environment var.

    TODO: make into a pure function and take ENV as input
    """

    if os.getenv("ENV") == "topic_model":
        bucket = os.getenv("WORD2VEC_BINARY_BUCKET")
        model = KeyedVectors.load_word2vec_format(
            f"s3://{bucket}/{filename}", binary=True
        )

    else:
        full_path = Path(__file__).parents[4] / "data" / folder / filename
        model = KeyedVectors.load_word2vec_format(
            full_path,
            binary=True,
        )

    return model


def model_search_word(
    word,
    model,
    glossary,
    hunspell=None,
    tags_guiding_typos_wv=None,
    return_spellcorrected_text=False,
):
    """
    Returns vector embedding (or None) for word, trying in order:
    1.  Special contextualization glossary (custom WVs)
    2.  Given case
    3.  Lowercase
    4.  Title case
    5.  Various alternative spellings of word, using Hunspell
            The best alternative spelling is that with highest cosine similarity
            to any of the words in tags_guiding_typos. Alternative spellings are
            only considered if tags_guiding_typos is not None.

    Parameters
    ----------
    word : str
    model : Word2Vec model (or KeyedVectors) - MUST BE PRE-NORMALIZED!
    glossary : dict
        Custom contextualization glossary. Words to replace are keys; their values
        must be dictionaries with (replacement words : weight) as key-value pairs
    hunspell : Hunspell object
        Very expensive to load, so should use shared object
    tags_guiding_typos_wv : list
        Set of tag wvs to guide the correction of misspelled words.For misspelled words,
        the best alternative spelling is that with highest cosine similarity
        to any of the wvs in tags_guiding_typos_wvs.

        E.g. if tags_guiding_typos = {"pregnant", ...}, then "chils" --> "child"
        rather than "chills", even though both are same edit distance.

        Optional parameter. If None (or None equivalent), step 5 above is skipped.
    return_spellcorrected_text : boolean
        If True, returns tuple (vector embedding, corrected spelling/case of word used)
    """
    if word.lower() in glossary:
        sum_vector = np.zeros(model["test"].shape)
        components = glossary[word.lower()]

        for c in components:
            sum_vector += model[c] * components[c]

        return_vector = sum_vector / np.sqrt(np.dot(sum_vector, sum_vector))
        return_word = word.lower()
    elif word in model:
        return_vector = model[word]
        return_word = word
    elif word.lower() in model:
        return_vector = model[word.lower()]
        return_word = word.lower()
    elif word.title() in model:
        return_vector = model[word.title()]
        return_word = word.title()
    elif (not hunspell) or (not tags_guiding_typos_wv):
        return_vector = None
    else:
        suggestions = hunspell.suggest(word)

        best_cs = 0
        best_alt = None
        best_alt_wv = None

        for alt in suggestions:
            # Note that when we call model_search_word here, we don't allow for further
            # spelling correction (by not passing in tags_guiding_typos). This prevents
            # theoretically infinite recursion: imagine that there are two words,
            # "thisword" and "thatword," that are both valid English (and in Hunspell)
            # but not in the model. Hunspell would continually offer the other word
            # as a suggestion.

            # Calculate WV for alt / suggestion
            alt_wv = model_search_word(alt, model, glossary)

            # Need to check alt spelling in the model/glossary either
            if alt_wv is not None:
                # Take max CS between alt and (any of the tags_guiding_typos)
                alt_cs = max(np.dot(alt_wv, tag_wv) for tag_wv in tags_guiding_typos_wv)

                if alt_cs > best_cs:
                    best_cs = alt_cs
                    best_alt = alt
                    best_alt_wv = alt_wv

        return_vector = best_alt_wv
        return_word = best_alt

    if return_vector is None:
        return None
    elif return_spellcorrected_text:
        return return_vector, return_word
    else:
        return return_vector


def model_search(
    tokens,
    model,
    glossary,
    hunspell=None,
    tags_guiding_typos_wv=None,
    return_spellcorrected_text=False,
):
    """
    Returns list of vector embeddings corresponding to given tokens

    return_spellcorrected_text : boolean
        If True, returns tuple (list[vector embeddings], list[spell-corrected tokens])
    """

    result = [
        model_search_word(
            word,
            model,
            glossary,
            hunspell,
            tags_guiding_typos_wv,
            return_spellcorrected_text,
        )
        for word in tokens
    ]

    if return_spellcorrected_text:
        tokens_vectors = [r[0] for r in result if r is not None]
        tokens_words = [r[1] for r in result if r is not None]

        return tokens_vectors, tokens_words
    else:
        return [r for r in result if r is not None]


def _nearest_k_percent(list_of_wvs, target_wv, k, floor):
    """
    Returns an array of shape (n_vectors, n_dim_embeddings)
    which contains the k% of word vectors (or floor, whichever is larger)
    in list_of_wvs that have highest CS to target_wv

    Parameters
    ----------
    list_of_wvs : list of vectors (1-D arrays)
    target_wv : vector (1-D array)
    k : float
        Nearest % of word vectors to capture
    floor : int
        Minimum # of word vectors to capture

    Returns
    -------
    ndarray
        (n_vectors, n_dim_embeddings)
        which contains the k% of word vectors (or floor, whichever is larger)
        in list_of_wvs that have highest CS to target_wv
    """
    cs = [np.dot(wv, target_wv) for wv in list_of_wvs]

    # k%, or 3, whichever is greater
    n_top = np.max([floor, int(k / 100 * len(list_of_wvs))])
    # No more than the entirety of wvs!
    n_top = np.min([n_top, len(list_of_wvs)])

    indices = np.argsort(cs)[::-1][:n_top]

    return np.vstack(list_of_wvs)[indices]


def _nearest_k_percent_average(list_of_wvs, target_wv, k, floor):
    """
    Obtains the k% of word vectors in list_of_wvs
    that are most similar to target_wv, then takes the mean

    Parameters
    ----------
    list_of_wvs : list of vectors (1-D arrays)
    target_wv : vector (1-D array)
    k : float
        Nearest % of word vectors to capture
    floor : int
        Minimum # of word vectors to capture

    Returns
    -------
    vector (1-D array)
        Mean of k% of word vectors (or floor, whichever is larger)
        in list_of_wvs that have highest CS to target_wv

    Notes
    -----
    All vectors are expected to be already normalized
    """
    nearest_wvs = _nearest_k_percent(list_of_wvs, target_wv, k, floor)

    return np.mean(nearest_wvs, axis=0)


def cs_nearest_k_percent_average(list_of_wvs, target_wv, k, floor):
    """
    Returns the cosine similarity between target_wv and
    the average of the normalized k% of word vectors in
    list_of_wvs that are most similar to target_wv

    Parameters
    ----------
    list_of_wvs : list of vectors (1-D arrays)
    target_wv : vector (1-D array)
    k : float
        Nearest % of word vectors to capture
    floor : int
        Minimum # of word vectors to capture

    Returns
    -------
    float
        Cosine similarity between target_wv and
        the mean of the k% of word vectors (or floor, whichever is larger)
        in list_of_wvs that have highest CS to target_wv

    Notes
    -----
    All vectors are expected to be already normalized
    """
    avg = _nearest_k_percent_average(list_of_wvs, target_wv, k, floor)
    avg_dot = np.dot(avg, avg)

    if np.isclose(avg_dot, 0):
        return 0
    else:
        return np.dot(avg, target_wv) / np.sqrt(avg_dot)
