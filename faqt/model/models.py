import numpy as np
from ..scoring_functions import cs_nearest_k_percent_average
from .embeddings import model_search_word, model_search


class FAQScorer:
    """
    Allows fitting FAQs and scoring new messages against it

    Parameters
    ----------
    w2v_model: gensim.models.KeyedVectors
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

    n_top_matches: int, optional
        The maximum number of matches to return. Should be provided to either
        the constructor (here) or when calling the `.score` method

    scoring_function: Callable[List[Array], Array[1d]] -> float, optional
        A function that takes a list of word vectors (incoming message) as the first
        argument and a vector (tag) as the second argument and returns a similarity
        score as a scalar float.
        - If not provided, used the `scoring_function.cs_nearest_k_percent_average`
        function.
        - If `None`, then it must be provided when calling the `score` method.

        Note: Additional arguments can be passed through `scoring_func_kwargs`

    **scoring_func_kwargs: dict, optional
        Additional arguments to be passed to the `scoring_function`.
        These can also be provided when calling the `.score` method


    Notes
    -----
    * w2v binary must contain prenormalized vectors. This is to reduce operations
      needed when calculating distance. See script in `faqt.scripts` to prenormalize
      your vectors.
    * The following parameters must be passed either in the constructor (here) or in
      the `.score()` method: `n_top_matches`, `scoring_function`,
      `**scoring_func_kwargs`.
    """

    def __init__(
        self,
        w2v_model,
        glossary=None,
        hunspell=None,
        tags_guiding_typos=None,
        n_top_matches=None,
        scoring_function=cs_nearest_k_percent_average,
        **scoring_func_kwargs
    ):
        self.w2v_model = w2v_model

        if glossary is None:
            glossary = {}

        if tags_guiding_typos is None:
            tags_guiding_typos = {}

        self.glossary = glossary
        self.hunspell = hunspell
        self.tags_guiding_typos = tags_guiding_typos
        self.n_top_matches = n_top_matches
        self.scoring_function = scoring_function
        self.scoring_func_kwargs = scoring_func_kwargs

        if self.tags_guiding_typos is not None:
            self.tags_guiding_typos_wv = model_search(
                tags_guiding_typos, w2v_model, glossary
            )
        else:
            self.tags_guiding_typos_wv = None

    def model_search_word(self, word):
        """
        Wrapper around embeddings.model_search_word. Sets the model and
        glossary and object attributes
        """
        return model_search_word(word, self.w2v_model, self.glossary)

    def model_search(self, message):
        """
        Wrapper around embeddings.model_search. Sets other arguments to object
        attributes
        """

        return model_search(
            message,
            model=self.w2v_model,
            glossary=self.glossary,
            hunspell=self.hunspell,
            tags_guiding_typos_wv=self.tags_guiding_typos_wv,
            return_spellcorrected_text=True,
        )

    def fit(self, faqs):
        """
        Fit FAQs

        #TODO: Define FAQ type and check that object is FAQ-like.

        Parameters
        ----------
        faqs: List[FAQ]
            List of FAQ-like objects.

        Returns
        -------
        Fitted FAQScorer

        """
        for faq in faqs:
            faq.faq_tags_wvs = {
                tag: self.model_search_word(tag) for tag in faq.faq_tags
            }
        self.faqs = faqs

        return self

    def score(
        self, message, n_top_matches=None, scoring_function=None, **scoring_func_kwargs
    ):
        """
        Scores a gives message and returns matches from FAQs.

        Parameters
        ----------
        message: List[str]
            pre-processed input message as a list of tokens.
            See `faqt.preprocessing` for preprocessing functions

        n_top_matches: int, optional
            The maximum number of matches to return. Should be provided to either
            the constructor or when calling the `.score` (here) method

        scoring_function: Callable[List[Array], Array[1d]] -> float, optional
            A function that takes a list of word vectors (incoming message) as the first
            argument and a vector (tag) as the second argument and returns a similarity
            score as a scalar float.
            - If `None`, use the `scoring_function` passed in the contructor.
            - If `None` and no `scoring_function` passed in contstructor, raise an
              exception.

            Note: Additional arguments can be passed through `scoring_func_kwargs`

        **scoring_func_kwargs: dict, optional
            Additional arguments to be passed to the `scoring_function`.
            These can also be provided when calling the `.score` method
            Returns
            -------
            Tuple[List[Tuple[Str, Str]], Dict, List[Str]]
                First item is a list of (FAQ id, FAQ content) tuples. This will have a max
                size of `n_top_matches`
                Second item is a Dictionary that shows the scores for each of the FAQs
                Third item is the spell corrected tokens for `message`.
        """
        if not hasattr(self, "faqs"):
            raise RuntimeError(
                "Model has not been fit. Please run .fit() method before .score"
            )
        scoring_function = self._get_updated_scoring_func(scoring_function)
        scoring_func_kwargs = self._get_updated_scoring_func_args(**scoring_func_kwargs)
        n_top_matches = self._get_n_top_matches(n_top_matches)

        top_matches_list = []
        scoring = {}
        inbound_vectors, inbound_spellcorrected = self.model_search(message)

        if len(inbound_vectors) == 0:
            return top_matches_list, scoring, ""

        scoring = get_faq_scores_for_message(
            inbound_vectors, self.faqs, scoring_function, **scoring_func_kwargs
        )
        top_matches_list = get_top_n_matches(scoring, n_top_matches)

        return top_matches_list, scoring, inbound_spellcorrected

    def _get_updated_scoring_func(self, my_scoring_func):
        """
        Resolve the scoring function to use. If no scoring function
        was passed in constructor or `.score()` method then raises an exception
        """

        if my_scoring_func is None:
            if self.scoring_function is None:
                raise ValueError(
                    (
                        "Must provide `scoring_function` either at init "
                        "or when calling `.score()`"
                    )
                )
            else:
                scoring_function = self.scoring_function
        else:
            scoring_function = my_scoring_func

        return scoring_function

    def _get_updated_scoring_func_args(self, **my_scoring_func_kwargs):
        """
        Updates scoring function arguments passed in constructor with
        arguments passed in `.score()`
        """

        scoring_func_kwargs = self.scoring_func_kwargs.copy()
        scoring_func_kwargs.update(my_scoring_func_kwargs)

        return scoring_func_kwargs

    def _get_n_top_matches(self, n_top_matches):
        """
        Get `n_top_matches` by checking argument passed to
        (1) `.score()` (2) the constructor
        """
        if n_top_matches is None:
            if self.n_top_matches is None:
                raise ValueError(
                    (
                        "`n_top_matches` must be passed either to constructor or "
                        "the `.score()` method"
                    )
                )
            else:
                n_top_matches = self.n_top_matches

        return n_top_matches


def get_faq_scores_for_message(
    inbound_vectors, faqs, scoring_function, **scoring_func_kwargs
):
    """
    Returns scores for the inbound vectors against each faq

    Parameters
    ----------
    inbound_vectors: List[Array]
        List of inbound tokens as word vectors
    faqs: List[FAQ]
        A list of faq-like objects. Each FAQ object must contain word vectors for
        each tags as a dictionary under `FAQ.faq_tags_wvs`

    Returns
    -------
    Dict[int, Dict]
        A Dictionary with `faq_id` as key. Values: faq details including scores
        for each tag and an `overall_score`
    """

    scoring = {}
    for faq in faqs:
        scoring[faq.faq_id] = {}
        scoring[faq.faq_id]["faq_title"] = faq.faq_title
        scoring[faq.faq_id]["faq_content_to_send"] = faq.faq_content_to_send
        scoring[faq.faq_id]["tag_cs"] = {}

        for tag, tag_wv in faq.faq_tags_wvs.items():
            if tag_wv is not None:
                scoring[faq.faq_id]["tag_cs"][tag] = scoring_function(
                    inbound_vectors, tag_wv, **scoring_func_kwargs
                )
            else:
                scoring[faq.faq_id]["tag_cs"][tag] = 0

        cs_values = list(scoring[faq.faq_id]["tag_cs"].values())
        scoring[faq.faq_id]["overall_score"] = (min(cs_values) + np.mean(cs_values)) / 2

    return scoring


def get_top_n_matches(scoring, n_top_matches):
    """
    Gives a list of scores for each FAQ, return the top `n_top_matches` FAQs

    Parameters
    ----------
    scoring: Dict[int, Dict]
        Dict with faq_id as key and faq details and scores as values.
        See return value of `get_faq_scores_for_message`.
    n_top_matches: int
        the number of top matches to return

    Returns
    -------
    List[Tuple(int, str)]
        A list of tuples of (faq_id, faq_content_to_send)._

    """
    matched_faq_titles = set()
    # Sort and copy over top matches
    top_matches_list = []
    for id in sorted(scoring, key=lambda x: scoring[x]["overall_score"], reverse=True):
        if scoring[id]["faq_title"] not in matched_faq_titles:
            top_matches_list.append(
                (scoring[id]["faq_title"], scoring[id]["faq_content_to_send"])
            )
            matched_faq_titles.add(scoring[id]["faq_title"])

        if len(matched_faq_titles) == n_top_matches:
            break
    return top_matches_list
