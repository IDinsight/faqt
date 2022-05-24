from ..scoring_functions import cs_nearest_k_percent_average
from .embeddings import model_search_word, model_search
from warnings import warn


class KeyedVectorsScorer:
    """
    Allows setting reference tagsets and scoring new messages against it

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
        scoring_func_kwargs={},
        summary_function_kwargs={},
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
        Scores a gives message and returns matches from tagset.

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
