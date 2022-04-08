import numpy as np
from ..scoring_functions import cs_nearest_k_percent_average
from .embeddings import model_search_word, model_search
from ..utils import AttributeDict
from warnings import warn


class KeyedVectorsScorer:
    """
    Allows setting reference FAQs and scoring new messages against it

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
    * w2v binary must contain prenormalized vectors. This is to reduce operations
      needed when calculating distance. See script in `faqt.scripts` to prenormalize
      your vectors.
    """

    def __init__(
        self,
        w2v_model,
        glossary=None,
        hunspell=None,
        tags_guiding_typos=None,
        n_top_matches=3,
        scoring_function=cs_nearest_k_percent_average,
        **scoring_func_kwargs,
    ):
        self.w2v_model = w2v_model

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

    def set_tags(self, tagset):
        """
        Set the tagset that messages should be matched against

        #TODO: Define tagset type and check that object is tagset-like.

        Parameters
        ----------
        tagset: List[DataClass]
            List of DataClass-like objects.
            Must have attribute containing 'tag'
            Tags are used to match incoming messages


        Returns
        -------
        self
        """
        if len(tagset) != 0:
            if isinstance(tagset[0], dict):
                tagset = list(map(lambda x: AttributeDict(x), tagset))

            tag_attribute = [x for x in dir(tagset[0]) if "tag" in x][0]

            for tags in tagset:
                tags.tags_wvs = {}
                for tag in getattr(tags, tag_attribute):
                    tag_wv = self.model_search_word(tag)
                    if tag_wv is None:
                        warn(
                            f"`{tag}` not found in vocab",
                            RuntimeWarning,
                        )
                    else:
                        tags.tags_wvs[tag] = tag_wv

        self.tagset = tagset

        return self

    def score(self, message, summary_function):
        """
        Scores a gives message and returns matches from tagset.

        Parameters
        ----------
        message: List[str]
            pre-processed input message as a list of tokens.
            See `faqt.preprocessing` for preprocessing functions

        summary_function: Function
            Function for scoring

        Returns
        -------
        Tuple[List[Tuple[Str, Str]], Dict, List[Str]]
            First item is a list of (FAQ id, FAQ content) tuples. This will have a
            max size of `n_top_matches`
            Second item is a Dictionary that shows the scores for each of the tagset
            Third item is the spell corrected tokens for `message`.
        """
        if not hasattr(self, "tagset"):
            raise RuntimeError(
                (
                    "Reference tags have not been set. Please run .set_tags()"
                    "method before .score"
                )
            )
        scoring_function = self.scoring_function
        scoring_func_kwargs = self.scoring_func_kwargs

        scoring = {}
        inbound_vectors, inbound_spellcorrected = self.model_search(message)

        if len(inbound_vectors) == 0:
            return scoring, ""

        scoring = summary_function(
            inbound_vectors, self.tagset, scoring_function, **scoring_func_kwargs
        )

        return scoring, inbound_spellcorrected

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
        each tags as a dictionary under `FAQ.tags_wvs`

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

        for tag, tag_wv in faq.tags_wvs.items():
            if tag_wv is not None:
                scoring[faq.faq_id]["tag_cs"][tag] = scoring_function(
                    inbound_vectors, tag_wv, **scoring_func_kwargs
                )
            else:
                scoring[faq.faq_id]["tag_cs"][tag] = 0

        cs_values = list(scoring[faq.faq_id]["tag_cs"].values())
        scoring[faq.faq_id]["overall_score"] = (min(cs_values) + np.mean(cs_values)) / 2

    return scoring


def get_topic_scores_for_message(
    inbound_vectors, topics, scoring_function, **scoring_func_kwargs
):
    """
    Returns scores for the inbound vectors against each topic

    Parameters
    ----------
    inbound_vectors: List[Array]
        List of inbound tokens as word vectors
    topics: List[Topics]
        A list of Topic-like objects. Each Topic object must contain word vectors for
        each tags as a dictionary under `Topic.tags_wvs`

    Returns
    -------
    Dict[int, float]
        A Dictionary with `_id` as key and similarity score as value
    """

    scoring = {}
    for topic in topics:
        all_tag_scores = []
        for tag, tag_wv in topic.tags_wvs.items():
            if tag_wv is not None:
                tag_score = scoring_function(
                    inbound_vectors, tag_wv, **scoring_func_kwargs
                )
            else:
                tag_score = 0
            all_tag_scores.append(tag_score)

        if len(all_tag_scores) == 0:
            raise RuntimeError("None of the tags were found in vocab")

        scoring[topic._id] = (np.max(all_tag_scores) + np.mean(all_tag_scores)) / 2

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
