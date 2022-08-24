from warnings import warn

import numpy as np
from faqt.scoring.single_tag_scoring import cs_nearest_k_percent_average


class KeyedVectorsScorer:
    """
    Allows setting reference tagsets and scoring new messages against it

    Parameters
    ----------
    word_embedding_model: gensim.models.KeyedVectors
        Only google news word2vec binaries have been tested. Vectors must be
        pre-normalized. See Notes below.

    glossary: Dict[str, Dict[str, float]], optional
        Custom contextualization glossary. Words to replace are keys; their values
        must be dictionaries with (replacement words : weight) as key-value pairs

    hunspell: Hunspell, optional
        Optional instance of the hunspell spell checker.

    tags_guiding_typos: List[str], optional
        Set of tag wvs to guide the correction of misspelled words.
        For misspelled words, the best alternative spelling is that with highest cosine
        similarity to any of the wvs in tags_guiding_typos_wvs.

        E.g. if tags_guiding_typos = {"pregnant", ...}, then "chils" --> "child"
        rather than "chills", even though both are same edit distance.

        Optional parameter. If None (or None equivalent), we assume not no guiding tags'

    n_top_matches: int
        The maximum number of matches to return.

    scoring_function: Callable[List[Array], Array[1d]] -> float, optional
        A function that takes a list of word vectors (incoming message) as the first
        argument and a vector (tag) as the second argument and returns a similarity
        score as a scalar float.
        Note: Additional arguments can be passed through `scoring_func_kwargs`

    **scoring_func_kwargs: dict, optional
        Additional arguments to be passed to the `scoring_function`.

    Notes
    -----
    * word embedding binary must contain prenormalized vectors. This is to
      reduce operations needed when calculating distance. See script in
      `faqt.scripts` to prenormalize your vectors.
    * A tagset is a collection of words (tags) designated to match incoming
      messages
    """

    def __init__(
        self,
        word_embedding_model,
        glossary=None,
        hunspell=None,
        tags_guiding_typos=None,
        n_top_matches=3,
        scoring_function=cs_nearest_k_percent_average,
        scoring_func_kwargs={},
    ):
        """Initialize"""
        self.word_embedding_model = word_embedding_model

        if glossary is None:
            glossary = {}

        if tags_guiding_typos is None:
            tags_guiding_typos = {}

        self.glossary = glossary.copy()
        self.hunspell = hunspell
        self.tags_guiding_typos = tags_guiding_typos.copy()
        self.n_top_matches = n_top_matches
        self.scoring_function = scoring_function
        self.scoring_func_kwargs = scoring_func_kwargs

        if self.tags_guiding_typos is not None:
            self.tags_guiding_typos_wv = model_search(
                tags_guiding_typos, word_embedding_model, glossary
            )
        else:
            self.tags_guiding_typos_wv = None

    def model_search_word(self, word):
        """
        Wrapper around embeddings.model_search_word. Sets the model and
        glossary and object attributes
        """
        return model_search_word(word, self.word_embedding_model, self.glossary)

    def model_search(self, message):
        """
        Wrapper around embeddings.model_search. Sets other arguments to object
        attributes
        """

        return model_search(
            message,
            model=self.word_embedding_model,
            glossary=self.glossary,
            hunspell=self.hunspell,
            tags_guiding_typos_wv=self.tags_guiding_typos_wv,
            return_spellcorrected_text=True,
        )

    def fit(self, tagset):
        """
        Set the tagset that messages should be matched against

        A tagset is a collection of words (tags) designated to match incoming messages

        Parameters
        ----------
        tagset: List[List[str]]
            List of list-likes of tags
            Tags are used to match incoming messages

        Returns
        -------
        self
        """
        tagset_wvs = []
        if len(tagset) != 0:
            for tags in tagset:
                tags_wvs = {}
                for tag in tags:
                    tag_wv = self.model_search_word(tag)
                    if tag_wv is None:
                        warn(
                            f"`{tag}` not found in vocab",
                            RuntimeWarning,
                        )
                    else:
                        tags_wvs[tag] = tag_wv
                tagset_wvs.append(tags_wvs)

        self.tagset = tagset
        self.tagset_wvs = tagset_wvs

        return self

    def score(self, message):
        """
        TODO: NEW METHOD that returns a score for each tagset! Uses _score
        """
        pass

    def _score(self, message):
        """
        Scores a given message against tagsets.

        Parameters
        ----------
        message: List[str]
            pre-processed input message as a list of tokens.
            See `faqt.preprocessing` for preprocessing functions

        Returns
        -------
        Tuple[List[Dict], List[Str]]
            First item is a list of dictionaries of score assigned to each tag in each
            tagset. Second item is the spell corrected tokens for `message`.
        """
        if not hasattr(self, "tagset_wvs"):
            raise RuntimeError(
                (
                    "Reference tags have not been set. Please run .set_tags()"
                    "method before .score"
                )
            )
        scoring_function = self.scoring_function
        scoring_func_kwargs = self.scoring_func_kwargs

        scoring = []
        inbound_vectors, inbound_spellcorrected = self.model_search(message)

        if len(inbound_vectors) == 0:
            return scoring, ""

        for tags_wvs in self.tagset_wvs:
            all_tag_scores = {}
            for tag, tag_wv in tags_wvs.items():
                if tag_wv is not None:
                    all_tag_scores[tag] = scoring_function(
                        inbound_vectors, tag_wv, **scoring_func_kwargs
                    )
                else:
                    all_tag_scores[tag] = 0

            scoring.append(all_tag_scores)

        return scoring, inbound_spellcorrected


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
