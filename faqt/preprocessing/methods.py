"""
Text pre-processing for word embeddings
"""
__all__ = ["preprocess_text_for_word_embedding", "preprocess_text_for_keyword_rule"]

from itertools import chain

from faqt.preprocessing.text import process_urls, remove_punctuation
from faqt.preprocessing.tokens import connect_phrases, get_ngrams, remove_stop_words
from nltk.tokenize import word_tokenize


def preprocess_text_for_word_embedding(
    content, entities_dict, n_min_dashed_words_url, reincluded_stop_words=None
):
    """
    Preprocess raw text strings to approximate preprocessing that goes into
    Google News word embeddings:

    1. Remove punctuation
    2. Tokenize using NLTK word_tokenize
    3. Remove stopwords
    4. Concatenate multi-word entities already in Google News model
       (e.g., "Niger_Delta")

    The following conventional text pre-processing steps are not done:
    - Lowercase
    - Lemmatization

    Parameters
    ----------
    content : str
        Original raw WhatsApp tokens
    entities_dict : Dict[Tuple[str], str]
        Example: entities_dict[('African', 'Union')] = "African_Union"
    n_min_dashed_words_url : Int
        The number of words that must be separated by dashes in a URL, to treat the
        text as an actual relevant content summary
    reincluded_stop_words: List[str], optional
        A list of words that should not be stop words.

    Returns
    -------
    tokens : list of str
        Pre-processed text as list of tokens.
    """

    content = process_urls(content, n_min_dashed_words_url)
    content = remove_punctuation(content)
    tokens = word_tokenize(content)

    tokens = remove_stop_words(tokens, reincluded_stop_words=reincluded_stop_words)
    tokens = connect_phrases(tokens, entities_dict)

    return tokens


def preprocess_text_for_keyword_rule(
    content,
    n_min_dashed_words_url,
    stem_func,
    spell_checker,
    reincluded_stop_words=None,
    ngram_min=1,
    ngram_max=2,
):
    """
    Preprocess raw text strings for keyword search:

    1. Remove punctuation
    2. Tokenize using NLTK word_tokenize
    3. Lowercase
    4. Spell correction
    5. Stem
    6. Extract ngrams

    Parameters
    ----------
    content : str
        Original raw WhatsApp tokens
    n_min_dashed_words_url : Int
        The number of words that must be separated by dashes in a URL, to treat the
        text as an actual relevant content summary
    stem_func: Callable
        A function that stems a given word
    spell_checker : hunspell.Hunspell instance or object
        A `hunspell.Hunspell` instance or an object with `spell` method which
        returns True for correctly spelled words, and `suggest` method which
        returns a tuple of suggested spell corrections. Pass an instance of
        `faqt.preprocessing.tokens.CustomHunspell` class to add custom words
        and spell corrections on top of the default hunspell class.
    reincluded_stop_words: List[str], optional
        A list of words that should not be stop words.
    ngram_min: int, default: 1
        Minimum for ngram range. To include single-gram tokens, set this to 1.
    ngram_max: int, default; 2
        Maximum for ngram range

    Returns
    -------
    tokens : list of str
        Pre-processed text as list of tokens.
    """
    assert ngram_max >= ngram_min

    content = content.lower()
    content = process_urls(content, n_min_dashed_words_url=n_min_dashed_words_url)
    content = remove_punctuation(content)

    tokens = word_tokenize(content)
    tokens = remove_stop_words(tokens, reincluded_stop_words=reincluded_stop_words)

    def spell_check_or_suggest(x):
        """checks spell and if wrong, suggest options"""
        if spell_checker.spell(x):
            return x
        else:
            suggestions = spell_checker.suggest(x)

            if len(suggestions) == 0:
                return x
            else:
                return suggestions[0]

    tokens = [spell_check_or_suggest(t) for t in tokens]

    # Tokenize and remove stop words for the second time, in case spell
    # correction generated new tokens / stop words.
    tokens = list(chain(*[word_tokenize(t) for t in tokens]))
    tokens = remove_stop_words(tokens, reincluded_stop_words=reincluded_stop_words)

    tokens = [stem_func(t) for t in tokens]

    ngram_tokens = get_ngrams(tokens, ngram_min, ngram_max)

    combined_ngram_tokens = list(map(lambda x: "_".join(x), ngram_tokens))
    return combined_ngram_tokens
