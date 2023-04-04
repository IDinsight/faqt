"""
Preprocessing methods that operate on a token or list of tokens.
"""

from copy import copy
from itertools import chain, tee

import hunspell

try:
    from hunspell import Hunspell
except ImportError:
    _has_hunspell = False
else:
    _has_hunspell = True
from nltk.corpus import stopwords


def _get_ngrams(tokens, n):
    """Return an iterator of ngram tuples from tokens:

    get_ngram_iterator('ABCD', 2) --> [('A', 'B'), ('B', 'C'), ('C', 'D')]
    get_ngram_iterator('ABCD', 3) --> [('A', 'B', 'C'), ('B', 'C', 'D')]
    get_ngram_iterator('A', 2) --> []

    """
    token_iterables = tee(tokens, n)

    for i in range(n):
        token_iterable = token_iterables[i]
        for j in range(i):
            next(token_iterable, None)
    return zip(*token_iterables)


def get_ngrams(tokens, ngram_min, ngram_max):
    """Return a list of ngrams from the given list of tokens"""
    assert ngram_min > 0
    assert ngram_max > 0
    assert ngram_max >= ngram_min

    mapped = map(lambda n: _get_ngrams(tokens, n), range(ngram_min, ngram_max + 1))
    chained = chain(*mapped)
    chained_list = list(chained)
    return chained_list


def connect_phrases(tokens, entities_dict):
    """
    Connect two or three word phrases with underscores,
    e.g. "African Union" => "African_Union"

    Parameters
    ----------
    tokens : list[str]
    entities_dict : dict[tuple, str]
        Example: entities_dict[('african', 'union')] = "African_Union"

    Returns
    -------
    tokens_connected : list of str
        List of tokens, with entities connected.
    """
    _tokens = [t.lower() for t in tokens]
    tokens_connected = copy(tokens)

    for n in [2, 3]:
        for i, ngram in enumerate(_get_ngrams(_tokens, n)):
            if ngram in entities_dict:
                tokens_to_replace = [entities_dict[ngram]] + ([None] * (n - 1))
                tokens_connected[i : i + n] = tokens_to_replace

    tokens_connected = list(filter(None, tokens_connected))
    return tokens_connected


def remove_stop_words(tokens, reincluded_stop_words=None):
    """Remove stop words in the list of tokens, except for re-included stop
    words."""
    if reincluded_stop_words is None:
        my_stop_words = set(stopwords.words("english"))
    else:
        my_stop_words = set(stopwords.words("english")) - set(reincluded_stop_words)

    return [t for t in tokens if t.lower() not in my_stop_words]


def check_gibberish(tokens):
    """Checks if the list of tokens constitute gibberish or not. A list of tokens is
    considered gibberish if
    1. they are all numbers, OR
    2. they are all misspelled in English.

    Parameters
    ----------
    tokens : List[str]
        List of tokens/words

    Returns
    -------
    boolean
        True if the list of tokens is gibberish, False otherwise.
    """
    all_numeric = all(t.isnumeric() for t in tokens)

    if all_numeric:
        return True

    if _has_hunspell:
        spell_checker = Hunspell()
        all_misspelled = all(not spell_checker.spell(t.lower()) for t in tokens)

        if all_misspelled:
            return True

    return False


class CustomHunspell(object):
    """
    ``hunspell.Hunspell``-like class with custom dictionary and custom spell
    correction. The following three items override the default Hunspell
    instance's ``spell`` and ``suggest`` methods.

    * Spell checking
        * ``custom_spell_check_list``: list of additional valid words.
    * Spell correction
        * ``custom_spell_correct_map``: Dictionary of custom spell corrections,
          e.g. ``{'jondis': 'jaundice'}`` to override ``hunspell.Hunspell.suggest`` spell
          correction method
        * ``priority_words``: List of words to consider first in a list of suggested
          spell-corrected words, in order of preference.

    Parameters
    ----------
    custom_spell_check_list : List[str], optional
        List of words to override ``hunspell.Hunspell.spell`` spell checker
    custom_spell_correct_map : Dict[str, str], optional
        Dictionary of custom spell corrections, e.g. ``{'jondis': 'jaundice'}``
        to override ``hunspell.Hunspell.suggest`` spell correction method
    priority_words : List[str]
        List of words to consider first in a list of suggested
        spell-corrected words, in order of preference.
    hunspell : hunspell.HunspellWrap, optional
        Optional Hunspell instance. If not provided, faqt creates a default
        Hunspell instance. (Hence, hunspell must be installed)

    Raises
    ------
    ImportError: if hunspell is not provided AND hunspell library is not installed.
    """

    def __init__(
        self,
        custom_spell_check_list=None,
        custom_spell_correct_map=None,
        priority_words=None,
        hunspell=None,
    ):
        """See class docstring for details."""
        if hunspell is None:
            if not _has_hunspell:
                raise ImportError(
                    f"Could not import hunspell library. If a hunspell object isn't "
                    f"passed, {self.__class__.__name__} requires hunspell library to instantiate a default Hunspell object."
                )

            self._hunspell = Hunspell()
        else:
            self._hunspell = hunspell

        if custom_spell_check_list is not None:
            for word in custom_spell_check_list:
                self._hunspell.add(word)

        self.custom_spell_check_list = custom_spell_check_list
        self.custom_spell_correct_map = custom_spell_correct_map or dict()
        self.priority_words = priority_words or list()

    def spell(self, token):
        """
        Check if token is a valid word or not.

        Parameters
        ----------
        token : str

        Returns
        -------
        bool
            True if it is a word in the saved hunspell dictionary (which has
            been updated with any custom words provided in `custom_spell_check_list`)
        """
        return self._hunspell.spell(token)

    def suggest(self, token):
        """
        Suggest spell corrections based on custom spell correction map,
        if it exists, otherwise using the `hunspell.suggest` method.

        Parameters
        ----------
        token : str

        Returns
        -------
        Tuple[str]
            Suggestions, in order of preference. The return type is consistent
            with the `hunspell.suggest` method.
        """
        corrected_token = self.custom_spell_correct_map.get(token, None)

        if corrected_token is not None:
            return (corrected_token,)

        suggestions = self._hunspell.suggest(token)

        if len(suggestions) > 0:
            for word in self.priority_words:
                if word in suggestions:
                    return (word,)

        return suggestions
