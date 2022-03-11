"""
Class for matching messages to topic
"""

import numpy as np
from ..scoring_functions import cs_nearest_k_percent_average
from .embeddings import model_search_word, model_search
from warnings import warn


class TopicModelScorer:
    """
    Allows scoring a message against various topics

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

    scoring_function: Callable[List[Array], Array[1d]] -> float, optional
        A function that takes a list of word vectors (incoming message) as the first
        argument and a vector (tag) as the second argument and returns a similarity
        score as a scalar float.
        If not provided, uses the `scoring_function.cs_nearest_k_percent_average`
        function.

        Note: Additional arguments can be passed through `scoring_func_kwargs`

    **scoring_func_kwargs: dict, optional
        Additional arguments to be passed to the `scoring_function`.

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
        scoring_function=cs_nearest_k_percent_average,
        **scoring_func_kwargs,
    ):
        self.w2v_model = w2v_model

        if glossary is None:
            glossary = {}

        if tags_guiding_typos is None:
            tags_guiding_typos = {}

        self.glossary = glossary
        self.hunspell = hunspell
        self.tags_guiding_typos = tags_guiding_typos
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

    def set_tags(self, topics):
        """
        Set the reference tags for topics. These are the tags that messages will be
        compared against.

        #TODO: Define a Topic type and check that object is Topic-like.

        Parameters
        ----------
        topics: List[Topic]
            List of Topic-like objects.

        Returns
        -------
        Fitted TopicModelScorer

        """
        for topic in topics:
            tag_wv_dict = {}
            for tag in topic.tags:
                tag_wv = self.model_search_word(tag)
                if tag_wv is None:
                    warn(
                        f"`{tag}` for topic {topic._id} not found in vocab",
                        RuntimeWarning,
                    )
                else:
                    tag_wv_dict[tag] = tag_wv
            topic.tags_wvs = tag_wv_dict
        self.topics = topics

        return self

    def score(self, message, scoring_function=None, **scoring_func_kwargs):
        """
        Scores a gives message and returns matches from Topics.

        Parameters
        ----------
        message: List[str]
            pre-processed input message as a list of tokens.
            See `faqt.preprocessing` for preprocessing functions

        Returns
        -------
        Tuple[Dict[int, float], List[Str]]
            First item is a Dictionary that shows the scores for each of the topics
            Second item is the spell corrected tokens for `message`.
        """
        if not hasattr(self, "topics"):
            raise RuntimeError(
                (
                    "Topic tags have not been set. Please run .set_tags() method "
                    "before .score"
                )
            )
        scoring_function = self.scoring_function
        scoring_func_kwargs = self.scoring_func_kwargs

        scoring = {}
        inbound_vectors, inbound_spellcorrected = self.model_search(message)

        if len(inbound_vectors) == 0:
            return scoring, ""

        scoring = get_scores_for_message(
            inbound_vectors, self.topics, scoring_function, **scoring_func_kwargs
        )

        return scoring, inbound_spellcorrected


def get_scores_for_message(
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
