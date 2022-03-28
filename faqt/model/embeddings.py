import numpy as np


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
