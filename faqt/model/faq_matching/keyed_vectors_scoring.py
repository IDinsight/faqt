import os
from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
from faqt.scoring.score_reduction import SCORE_REDUCTION_METHODS
from faqt.scoring.score_weighting import SCORE_WEIGHTING_METHODS
from faqt.scoring.tag_scoring import TAG_SCORING_METHODS
from gensim.similarities import WmdSimilarity
from nltk.tokenize import word_tokenize


class KeyedVectorsScorerBase(ABC):
    """
    Base class for Keyed vector scoring-type models

    Allows setting reference contents and scoring new messages against it

    Parameters
    ----------
    word_embedding_model: gensim.models.KeyedVectors
        Only google news word2vec binaries have been tested. Vectors must be
        pre-normalized. See Notes below.

    tokenizer: Callable[[str], list[str]], optional
        Tokenizer for input message and/or contents. May include preprocessing
        steps. If None, uses ``nltk.tokenize.word_tokenize`` from the `nltk` library.

    glossary: Dict[str, Dict[str, float]], optional
        Custom contextualization glossary. Words to replace are keys; their
        values must be dictionaries with (replacement words : weight) as key-value
        pairs. All keys will be turned into its lowercase form. When looking up a word
        in the glossary, the lowercased token will be looked up. If a token is not found
        in the glossary, it will be left as is.

    hunspell: Hunspell, optional
        Optional instance of the hunspell spell checker.

    tags_guiding_typos: List[str], optional
        Set of tag wvs to guide the correction of misspelled words.
        For misspelled words, the best alternative spelling is that with
        highest cosine
        similarity to any of the wvs in tags_guiding_typos_wvs.

        E.g. if ``tags_guiding_typos = {"pregnant", ...}``, then "chils" -->
        "child"
        rather than "chills", even though both are same edit distance.

        Optional parameter. If None (or None equivalent), we assume not no
        guiding tags'

    weighting_method : string or Callable, optional
        If a string is passed, must be in
        ``faqt.scoring.score_weighting.SCORE_WEIGHTING_METHODS``,
        e.g. ``add_weights``.

        If a Callable is passed, it must be a function that takes in two
        lists of floats (scores and weights) with optional
        keyword-arguments and outputs a 1-D array of weighted scores.

    weighting_kwargs : dict, optional
        Keyword arguments to pass to ``weighting_method``

    Notes
    -----
    * A tagsets is a collection of words (tags) designated to match incoming
      messages
    """

    def __init__(
        self,
        word_embedding_model,
        tokenizer=None,
        glossary=None,
        hunspell=None,
        tags_guiding_typos=None,
        weighting_method=None,
        weighting_kwargs=None,
    ):
        """Initialize"""
        self.word_embedding_model = word_embedding_model

        glossary = glossary or {}
        tags_guiding_typos = tags_guiding_typos or {}

        if tokenizer is None:
            tokenizer = word_tokenize

        self.tokenizer = tokenizer
        self.glossary = preprocess_glossary(glossary)
        self.hunspell = hunspell
        self.tags_guiding_typos = tags_guiding_typos.copy()

        if self.tags_guiding_typos is not None:
            self.tags_guiding_typos_wv = model_search(
                tags_guiding_typos, word_embedding_model, glossary
            )
        else:
            self.tags_guiding_typos_wv = None

        if isinstance(weighting_method, str):
            weighting_method = SCORE_WEIGHTING_METHODS[weighting_method]

        self.weighting_method = weighting_method
        self.weighting_kwargs = weighting_kwargs

        self.contents = None
        self.content_vectors = None
        self.content_weights = None

    def set_glossary(self, glossary):
        """Set glossary"""
        self.glossary = preprocess_glossary(glossary)

    def set_tokenizer(self, tokenizer):
        """Set tokenizer"""
        self.tokenizer = tokenizer

    def set_tags_guiding_typos(self, tags_guiding_typos):
        """Set tag guiding typos"""
        self.tags_guiding_typos = tags_guiding_typos

    @abstractmethod
    def set_contents(self, contents, weights=None):
        """Sets content: preprocesses `contents` and `weights` as necessary
        and saves them to `self.contents`, `self.content_weights` and
        optionally saves word-vectors to `self.content_vectors`."""
        raise NotImplementedError

    def score_contents(
        self, message, return_spell_corrected=False, weights=None, **kwargs
    ):
        """
        Scores contents and applies weighting if `self.weighting_method` is
        not None

        Parameters
        ----------
        message : str
        return_spell_corrected : bool, default=False
        weights: List[str] or None
            Weight of each FAQ, will override content_weights if added

        kwargs :
            additional keyword arguments to pass.
            e.g. for StepwiseKeyedVectorsScorer, `return_tag_scores=True` will
            include an extra item in the return dictionary, with key
            "tag_scores" and value a dictionary of tags with scores for each
            token in the message.

        Returns
        -------
        return_dict : dict
            `return_dict["overall_scores"]` : Score for each content in the
            order stored in `self.content`.
            `return_dict["spell_corrected"]` : List of spell-corrected
            pre-processed tokens from `message`, if `return_spell_corrected==True`
            Derived classes may return additional key-value pairs.
        """
        if not self.is_set:
            raise ValueError(
                "Contents have not been set. Set contents with " "`self.set_contents()`"
            )

        if weights is not None and self.content_weights is not None:
            warn(
                "`weights` parameter is provided. This will override the `content_weights` set during intialization. "
            )

        message_tokens = self.tokenizer(message)
        message_vectors, spell_corrected = self.model_search(message_tokens)

        if len(message_vectors) == 0:
            result = {"overall_scores": []}
            if return_spell_corrected:
                result["spell_corrected"] = []
            if kwargs.get("return_tag_scores"):
                result["tag_scores"] = []

            return result

        result = self._score_contents(message_vectors, spell_corrected, **kwargs)

        if self.weighting_method is not None and self.content_weights is not None:
            if weights:
                content_weights = weights
            else:
                content_weights = self.content_weights

            weighted_scores = self.weighting_method(
                result["overall_scores"], content_weights, **self.weighting_kwargs
            )

            result["overall_scores"] = weighted_scores

        if return_spell_corrected:
            result["spell_corrected"] = spell_corrected

        return result

    @abstractmethod
    def _score_contents(self, message_vectors, spell_corrected, **kwargs):
        """Abstract method to do the scoring on message vectors and/or spell
        corrected words, without weighting. Must be implemented for each
        child class."""
        raise NotImplementedError

    @property
    def is_set(self):
        """Check if the contents have been set, i.e. ready for scoring"""
        return self.contents is not None

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

    def _check_weight_inputs(self, weights):
        """Raises warning if there are inconsistencies with the weighting
        method parameter or how weights were previously set, if at all.
        Returns normalized weights."""
        if self.content_weights is not None and weights is None:
            warn(
                "You have previously set contents with `weights` that are "
                "not None, but are now setting the weights to be None."
            )

        if weights is None and self.weighting_method is not None:
            warn(
                f"Weighting method is set to {self.weighting_method} but "
                f"parameter `weights` was not provided. No weighting will be "
                f"performed!"
            )
        elif weights is not None:
            if self.weighting_method is None:
                warn(
                    "Weights were provided but `self.weighting_method` is "
                    "None. No weighting will be performed!"
                )

            if not np.isclose(np.sum(weights), 1.0):
                weights = weights / np.sum(weights)

        return weights


class StepwiseKeyedVectorsScorer(KeyedVectorsScorerBase):
    """
    Allows setting reference contents (as tagsets) and scoring new messages
    against it. Allows for token-level pairwise scoring,
    where ``tag_scoring_method`` is applied to score the message on
    individual tags first, then ``score_reduction_method`` is applied to
    reduce the scores to a single overall score for the tagset.

    Parameters
    ----------
    word_embedding_model: gensim.models.KeyedVectors
        Only google news word2vec binaries have been tested. Vectors must be
        pre-normalized. See Notes below.

    tag_scoring_method : string or Callable, default: ``"cs_nearest_k_percent_average"``
        If a string is passed, it must be in
        `faqt.scoring.tag_scoring.TAG_SCORING_METHODS`.

        If a callable is passed it must be a function that takes in a
        list of vectors (tokens tokens) as its
        first argument and a single vector (tag) as the second argument.

    tag_scoring_kwargs : dict, optional
        Keyword arguments for `tag_scoring_method`

    score_reduction_method : string or Callable, default: "simple_mean"
        If a string is passed it must be in
        `faqt.scoring.score_reduction.SCORE_REDUCTION_METHODS`.

        If a callable is passed it must be a function that takes in a
        list of floats (scores) and reduces it to a scalar float value.

    score_reduction_kwargs : dict, optional
        Keyword arguments for `score_reduction_method`

    weighting_method : string or Callable, optional
        If a string is passed, must be in
        `faqt.scoring.score_weighting.SCORE_WEIGHTING_METHODS`,
        e.g. `add_weights`.

        If a Callable is passed, it must be a function that takes in two
        lists of floats (scores and weights) with optional
        keyword-arguments and outputs a 1-D array of weighted scores.

    weighting_kwargs : dict, optional
        Keyword arguments to pass to `weighting_method`

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
        highest cosine similarity to any of the wvs in `tags_guiding_typos_wvs.`

        E.g. if ``tags_guiding_typos = {"pregnant", ...}``, then "chils" -->
        "child" rather than "chills", even though both are same edit
        distance.

        Optional parameter. If None (or None equivalent), we assume not no
        guiding tags'

    Notes
    -----
    * word embedding binary must contain prenormalized vectors. This is to
      reduce operations needed when calculating distance. See script in
      ``faqt.scripts`` to prenormalize your vectors.
    * A tagsets is a collection of words (tags) designated to match incoming
      messages
    """

    def __init__(
        self,
        word_embedding_model,
        tokenizer=None,
        tag_scoring_method="cs_nearest_k_percent_average",
        tag_scoring_kwargs=None,
        score_reduction_method="simple_mean",
        score_reduction_kwargs=None,
        weighting_method=None,
        weighting_kwargs=None,
        glossary=None,
        hunspell=None,
        tags_guiding_typos=None,
    ):
        """Initialize"""
        super(StepwiseKeyedVectorsScorer, self).__init__(
            word_embedding_model,
            tokenizer=tokenizer,
            glossary=glossary,
            hunspell=hunspell,
            tags_guiding_typos=tags_guiding_typos,
            weighting_method=weighting_method,
            weighting_kwargs=weighting_kwargs,
        )

        if isinstance(tag_scoring_method, str):
            tag_scoring_method = TAG_SCORING_METHODS[tag_scoring_method]
        self.tag_scoring_method = tag_scoring_method
        self.tag_scoring_kwargs = tag_scoring_kwargs or {}

        if isinstance(score_reduction_method, str):
            score_reduction_method = SCORE_REDUCTION_METHODS[score_reduction_method]
        self.score_reduction_method = score_reduction_method
        self.score_reduction_kwargs = score_reduction_kwargs or {}

    def set_contents(self, contents, weights=None):
        """
        Set the contents that messages should be matched against.

        For this class, each content must be (represented by) a tagset,
        which is a collection of words (tags) designated to match incoming
        messages.

        Parameters
        ----------
        contents: List[List[str]]
            A list of pre-defined list-likes of tags.
            Tags are used to match incoming messages
        weights: List[float] or None
            Weight of each FAQ, will be scaled so that sum(weights) == 1.0.

        Returns
        -------
        self
        """
        tagset_wvs = []

        if len(contents) != 0:
            for tag_set in contents:
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

        weights = self._check_weight_inputs(weights)

        self.contents = contents
        self.content_vectors = tagset_wvs
        self.content_weights = weights

        return self

    def _score_contents(
        self, message_vectors, spell_corrected_tokens=None, return_tag_scores=False
    ):
        """
        Scores the message on contents in steps.
            1. Score each (tag, message) pair
            2. Reduce scores from 1 for each tagset, so we get a score for
            each (tagset, message) pair.
        """

        overall_scores = []
        tag_scores = []

        for tag2vector in self.content_vectors:
            tag_scores_per_tagset = {}

            for tag, vector in tag2vector.items():
                tag_scores_per_tagset[tag] = self.tag_scoring_method(
                    message_vectors, vector, **self.tag_scoring_kwargs
                )
            tag_scores.append(tag_scores_per_tagset)

            overall_score = self.score_reduction_method(
                list(tag_scores_per_tagset.values()),
                **self.score_reduction_kwargs,
            )

            overall_scores.append(overall_score)

        return_dict = {"overall_scores": overall_scores}

        if return_tag_scores:
            return_dict["tag_scores"] = tag_scores

        return return_dict

    @property
    def is_set(self):
        """Check if the contents have been set, i.e. ready for scoring.
        Overrides base class' property since StepwiseKeyedVectorsScorer also
        requires `self.content_vectors` to be not None."""
        return self.contents is not None and self.content_vectors is not None


class WMDScorer(KeyedVectorsScorerBase):
    """WMD distance scoring model"""

    def __init__(
        self,
        word_embedding_model,
        tokenizer=None,
        weighting_method=None,
        weighting_kwargs=None,
        glossary=None,
        hunspell=None,
        tags_guiding_typos=None,
    ):
        """Initialize"""
        super(WMDScorer, self).__init__(
            word_embedding_model,
            tokenizer=tokenizer,
            weighting_method=weighting_method,
            weighting_kwargs=weighting_kwargs,
            glossary=glossary,
            hunspell=hunspell,
            tags_guiding_typos=tags_guiding_typos,
        )
        self.wmd_index = None

    def set_contents(self, contents, weights=None):
        """
        Set the contents that messages should be matched against.

        For this class, each message must be (represented by) a tagset,
        which is a collection of words (tags) designated to match incoming
        messages.

        Parameters
        ----------
        contents: List[str] or List[List[str]]
            List of contents. Accepts a list of un-tokenized strings as well
            as a list of tokens (list[list[str]]) for backwards compatibility.
            Support for list of list of tokens will be dropped in the future.
        weights: List[float] or None
            Weight of each content, will be scaled so that sum(weights) == 1.0.

        Returns
        -------
        self
        """
        preprocessed_content_tokens = []

        if all(isinstance(x, str) for x in contents):
            tokenize = True
        elif all(isinstance(x, list) for x in contents) and all(
            isinstance(y, str) for x in contents for y in x
        ):
            warn(
                "Support for list[list[str]] is for backwards compatibility and will "
                "be dropped in the future. Pass list[str] and FAQT will automatically "
                "tokenize each item using `self.tokenizer`.",
                DeprecationWarning,
            )
            tokenize = False
        else:
            raise TypeError("`contents` must be of type List[str] or List[List[str]].")

        for content in contents:
            if tokenize:
                tokens = self.tokenizer(content)
            else:
                tokens = content

            # gensim's WMD drops tokens that aren't in the dictionary, so we want to
            # make sure we can find the content tokens in the model
            _, tokens = self.model_search(tokens)
            preprocessed_content_tokens.append(tokens)

        self.wmd_index = WmdSimilarity(
            corpus=preprocessed_content_tokens,
            kv_model=self.word_embedding_model,
            chunksize=np.ceil(len(contents) / os.cpu_count()).astype(int),
        )
        weights = self._check_weight_inputs(weights)

        self.contents = preprocessed_content_tokens
        self.content_weights = weights

        return self

    @property
    def is_set(self):
        """Checks if contents are set. Overrides base class' property."""
        return self.contents is not None and self.wmd_index is not None

    def _score_contents(
        self,
        message_vectors,
        spell_corrected_tokens,
        return_spell_corrected=False,
        **kwargs,
    ):
        """
        Scores the message on each content using the word-mover's distance
        implementation from gensim.
        """

        scores = self.wmd_index[spell_corrected_tokens]

        result = {"overall_scores": scores}

        return result


def preprocess_glossary(glossary):
    """Preprocess glossary for use in `model_search_word` and `model_search`.
    Currently only lowercases keys.

    Parameters
    ----------
    glossary : Dict
        Glossary to preprocess.

    Returns
    -------
    Dict
        Preprocessed glossary
    """
    lowercased_keys = {k.lower(): v for k, v in glossary.items()}
    return lowercased_keys


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
    to any of the words in `tags_guiding_typos`. Alternative spellings are
    only considered if `tags_guiding_typos` is not None.

    Parameters
    ----------
    word : str
        Word to search in the model
    model : gensim.models.KeyedVectors
        If using Google News Embeddings, it must be pre-normalized.
    glossary : dict, optional
        Custom contextualization glossary. Words to replace are keys; their values
        must be dictionaries with (replacement words : weight) as key-value
        pairs. If None, no custom contextualization is performed.
    hunspell : Hunspell object
        Very expensive to load, so should use shared object
    tags_guiding_typos_wv : list
        Set of tag wvs to guide the correction of misspelled words.For misspelled words,
        the best alternative spelling is that with highest cosine similarity
        to any of the wvs in tags_guiding_typos_wvs.

        E.g. if ``tags_guiding_typos = {"pregnant", ...}``, then "chils" --> "child"
        rather than "chills", even though both are same edit distance.

        Optional parameter. If None (or None equivalent), step 5 above is skipped.
    return_spellcorrected_text : boolean
        If True, returns tuple (vector embedding, corrected spelling/case of word used)
    """
    if glossary is not None and word.lower() in glossary:
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
    glossary=None,
    hunspell=None,
    tags_guiding_typos_wv=None,
    return_spellcorrected_text=False,
):
    """
    Returns list of vector embeddings corresponding to given tokens. See
    `faqt.model.faq_matching.keyed_vector_scoring.model_search_word` for how
    exactly the tokens are searched in the model.

    Parameters
    ----------
    tokens : list[str]
        List of words to search from the model.
    model : gensim.models.KeyedVectors
        If using Google News Embeddings, it must be pre-normalized.
    glossary : dict or None
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

        Optional parameter.
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
