from warnings import warn

import numpy as np


class KeyedVectorsScorerBase:
    """Base class for Keyed vector scoring-type models"""

    def __init__(
        self,
        word_embedding_model,
        glossary=None,
        hunspell=None,
        tags_guiding_typos=None,
    ):
        """
        Allows setting reference tagsets and scoring new messages against it

        Parameters
        ----------
        word_embedding_model: gensim.models.KeyedVectors
            Only google news word2vec binaries have been tested. Vectors must be
            pre-normalized. See Notes below.

        glossary: Dict[str, Dict[str, float]], optional
            Custom contextualization glossary. Words to replace are keys; their
            values
            must be dictionaries with (replacement words : weight) as key-value
            pairs

        hunspell: Hunspell, optional
            Optional instance of the hunspell spell checker.

        tags_guiding_typos: List[str], optional
            Set of tag wvs to guide the correction of misspelled words.
            For misspelled words, the best alternative spelling is that with
            highest cosine
            similarity to any of the wvs in tags_guiding_typos_wvs.

            E.g. if tags_guiding_typos = {"pregnant", ...}, then "chils" --> "child"
            rather than "chills", even though both are same edit distance.

            Optional parameter. If None (or None equivalent), we assume not no
            guiding tags'
        Notes
        -----
        * word embedding binary must contain prenormalized vectors. This is to
          reduce operations needed when calculating distance. See script in
          `faqt.scripts` to prenormalize your vectors.
        * A tagsets is a collection of words (tags) designated to match incoming
          messages
        """

        self.word_embedding_model = word_embedding_model

        if glossary is None:
            glossary = {}

        if tags_guiding_typos is None:
            tags_guiding_typos = {}

        self.glossary = glossary.copy()
        self.hunspell = hunspell
        self.tags_guiding_typos = tags_guiding_typos.copy()

        if self.tags_guiding_typos is not None:
            self.tags_guiding_typos_wv = model_search(
                tags_guiding_typos, word_embedding_model, glossary
            )
        else:
            self.tags_guiding_typos_wv = None

    def set_contents(self, tagsets, weights=None):
        """
        Set the contents that messages should be matched against.

        For this class, each content must be (represented by) a tagset,
        which is a collection of words (tags) designated to match incoming
        messages.

        Parameters
        ----------
        tagsets: List[List[str]]
            List of list-likes of tags
            Tags are used to match incoming messages
        weights: List[float] or None
            Weight of each FAQ

        Returns
        -------
        self
        """
        tagset_wvs = []
        if len(tagsets) != 0:
            for tag_set in tagsets:
                tag2vector = {}
                for tag in tag_set:
                    word_vector = self.model_search_word(tag)
                    if word_vector is None:
                        warn(
                            f"`{tag}` not found in vocab",
                            RuntimeWarning,
                        )
                    else:
                        tag2vector[tag] = word_vector
                tagset_wvs.append(tag2vector)

        self.tagsets = tagsets
        self.tagset_wvs = tagset_wvs

        if not np.isclose(np.sum(weights), 1.0):
            weights = weights / np.sum(weights)
        self.tagset_weights = weights

        return self

    def model_search_word(self, word):
        """
        Wrapper around embeddings.model_search_word. Sets the model and
        glossary and object attributes
        """
        return model_search_word(word, self.word_embedding_model, self.glossary)

    def model_search(self, tokens):
        """
        Wrapper around embeddings.model_search. Sets other arguments to object
        attributes
        """

        return model_search(
            tokens,
            model=self.word_embedding_model,
            glossary=self.glossary,
            hunspell=self.hunspell,
            tags_guiding_typos_wv=self.tags_guiding_typos_wv,
            return_spellcorrected_text=True,
        )


class KeyedVectorsScorer(KeyedVectorsScorerBase):
    """Simple model"""

    def __init__(
        self,
        word_embedding_model,
        scoring_method,
        scoring_kwargs=None,
        glossary=None,
        hunspell=None,
        tags_guiding_typos=None,
        N=1.0,
    ):
        """
        Allows setting reference tagsets and scoring new messages against it

        Parameters
        ----------
        word_embedding_model: gensim.models.KeyedVectors
            Only google news word2vec binaries have been tested. Vectors must be
            pre-normalized. See Notes below.

        scoring_method: Callable
            A function that takes a pair of list of word vectors (e.g. word
            vectors of tokens in incoming tokens and word vectors of tags) and
            returns a match score as a scalar float.

            # TODO: determine if we should provide a default or not
            # If not provided, defaults to a function that first applies
            # `faqt.scoring.tag_scoring.cs_nearest_k_percent_average` to each tag (
            # with the tokens) and then reducing the scores using
            # `faqt.scoring.reduce.simple_mean`.

        scoring_kwargs : dict, optional
            Keyword arguments to pass through `scoring_method`

        glossary: Dict[str, Dict[str, float]], optional
            Custom contextualization glossary. Words to replace are keys; their
            values
            must be dictionaries with (replacement words : weight) as key-value
            pairs

        hunspell: Hunspell, optional
            Optional instance of the hunspell spell checker.

        tags_guiding_typos: List[str], optional
            Set of tag wvs to guide the correction of misspelled words.
            For misspelled words, the best alternative spelling is that with
            highest cosine
            similarity to any of the wvs in tags_guiding_typos_wvs.

            E.g. if tags_guiding_typos = {"pregnant", ...}, then "chils" --> "child"
            rather than "chills", even though both are same edit distance.

            Optional parameter. If None (or None equivalent), we assume not no
            guiding tags'

        N : float
            "strength" of weights -- this is the N in weighted final score:
            `(overall_score + N * weight) / (1 + N)`
            Default is 0.0 (no weight added)

        Notes
        -----
        * word embedding binary must contain prenormalized vectors. This is to
          reduce operations needed when calculating distance. See script in
          `faqt.scripts` to prenormalize your vectors.
        * A tagsets is a collection of words (tags) designated to match incoming
          messages
        """
        super(KeyedVectorsScorer, self).__init__(
            word_embedding_model, glossary, hunspell, tags_guiding_typos
        )

        self.scoring_method = scoring_method
        self.scoring_kwargs = scoring_kwargs or {}

        self.N = N

    def score_contents(self, message, return_corrected=False):
        """
        Scores each tagset against the given tokens
        Parameters
        ----------
        message: List[str]
            pre-processed input tokens as a list of tokens.
            See `faqt.preprocessing` for preprocessing functions
        return_corrected : bool
            If True, returns a list of spell-corrected tokens of `tokens`.

        Returns
        -------
        Tuple(List[float], [List[Dict[str, float]], List[Str], ])
            Score for each tag-set in the order stored in `self.tagsets`.
            Optionally also returns a list of dictionaries of score assigned
            to each tag in each tagset and/or a list of spell corrected tokens
            of `tokens`.

        """
        message_vectors, spell_corrected = self.model_search(message)

        overall_scores = []

        for tag_vectors in self.tagset_wvs:
            overall_score = self.scoring_method(
                message_vectors, tag_vectors.values(), **self.scoring_kwargs
            )

            overall_scores.append(overall_score)

        if self.tagset_weights is not None and self.N > 0.0:
            overall_scores = [
                (score + self.N * w) / (self.N + 1)
                for score, w in zip(overall_scores, self.tagset_weights)
            ]

        if return_corrected:
            return overall_scores, spell_corrected
        else:
            return overall_scores


class StepwiseKeyedVectorScorer(KeyedVectorsScorerBase):
    """Stepwise model"""

    def __init__(
        self,
        word_embedding_model,
        tag_scoring_method,
        score_reduction_method,
        tag_scoring_kwargs=None,
        score_reduction_kwargs=None,
        glossary=None,
        hunspell=None,
        tags_guiding_typos=None,
        N=1.0,
    ):
        """
        Allows setting reference contents (as tagsets) and scoring new messages
        against it. Allows for token-level pairwise scoring,
        where `tag_scoring_method` is applied to score the message on
        individual tags first, then `score_reduction_method` is applied to
        reduce the scores to a single overall score for the tagset.

        Parameters
        ----------
        word_embedding_model: gensim.models.KeyedVectors
            Only google news word2vec binaries have been tested. Vectors must be
            pre-normalized. See Notes below.

        tag_scoring_method : Callable
            A function that takes in a list of vectors (tokens tokens) as its
            first argument and a single vector (tag) as the second argument.

            See `faqt.scoring.tag_scoring` for pre-defined tag scoring methods.

        score_reduction_method : Callable
            A function that takes in a list of floats (scores) and reduces it to a scalar float value.
            See `faqt.scoring.reduce` for pre-defined score reduction methods.

        tag_scoring_kwargs : dict, optional
            Keyword arguments for `tag_scoring_method`

        score_reduction_kwargs : dict, optional
            Keyword arguments for `score_reduction_method`

        glossary: Dict[str, Dict[str, float]], optional
            Custom contextualization glossary. Words to replace are keys; their
            values
            must be dictionaries with (replacement words : weight) as key-value
            pairs

        hunspell: Hunspell, optional
            Optional instance of the hunspell spell checker.

        tags_guiding_typos: List[str], optional
            Set of tag wvs to guide the correction of misspelled words.
            For misspelled words, the best alternative spelling is that with
            highest cosine
            similarity to any of the wvs in tags_guiding_typos_wvs.

            E.g. if tags_guiding_typos = {"pregnant", ...}, then "chils" -->
            "child"
            rather than "chills", even though both are same edit distance.

            Optional parameter. If None (or None equivalent), we assume not no
            guiding tags'

        N : float
            "strength" of weights -- this is the N in weighted final score:
            `(overall_score + N * weight) / (1 + N)`
            Default is 0.0 (no weight added)

        Notes
        -----
        * word embedding binary must contain prenormalized vectors. This is to
          reduce operations needed when calculating distance. See script in
          `faqt.scripts` to prenormalize your vectors.
        * A tagsets is a collection of words (tags) designated to match incoming
          messages
        """
        super(StepwiseKeyedVectorScorer, self).__init__(
            word_embedding_model, glossary, hunspell, tags_guiding_typos
        )

        self.tag_scoring_method = tag_scoring_method
        self.tag_scoring_kwargs = tag_scoring_kwargs or {}

        self.score_reduction_method = score_reduction_method
        self.score_reduction_kwargs = score_reduction_kwargs or {}

        self.N = N

    def score_contents(self, message, return_tag_scores=False, return_corrected=False):
        """

        Parameters
        ----------
        message: List[str]
            pre-processed input tokens as a list of tokens.
            See `faqt.preprocessing` for preprocessing functions
        return_tag_scores : bool
            If True, returns a list of dictionaries, where each dictionary
            contains the scores for each tag.
        return_corrected : bool
            If True, returns a list of spell-corrected tokens of `tokens`.

        Returns
        -------
        Tuple(List[float], [List[Dict[str, float]], List[Str], ])
            Score for each tag-set in the order stored in `self.tagsets`.
            Optionally also returns a list of dictionaries of score assigned
            to each tag in each tagset and/or a list of spell corrected tokens
            of `tokens`.

        """
        message_vectors, spell_corrected = self.model_search(message)

        overall_scores = []
        tag_scores = []

        for tag2vector in self.tagset_wvs:
            tag_scores_per_tagset = {}

            for tag, vector in tag2vector.items():
                tag_scores_per_tagset[tag] = self.tag_scoring_method(
                    message_vectors, vector, **self.tag_scoring_kwargs
                )
            tag_scores.append(tag_scores_per_tagset)

            overall_score = self.score_reduction_method(
                message_vectors,
                tag_scores_per_tagset.values(),
                **self.score_reduction_kwargs,
            )

            overall_scores.append(overall_score)

        # TODO: to allow multiple weighting functions or not?
        if self.tagset_weights is not None and self.N > 0.0:
            overall_scores = [
                (score + self.N * w) / (self.N + 1)
                for score, w in zip(overall_scores, self.tagset_weights)
            ]

        if return_tag_scores:
            if return_corrected:
                return overall_scores, tag_scores, spell_corrected
            else:
                return overall_scores, tag_scores
        elif return_corrected:
            return overall_scores, spell_corrected
        return overall_scores


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
