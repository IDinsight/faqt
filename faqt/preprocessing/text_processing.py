"""
Text pre-processing for word embeddings
"""

import re
import urllib
from itertools import tee

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

__all__ = ["preprocess_text"]


def preprocess_text(
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
        Original raw WhatsApp message
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

    content = _process_urls(content, n_min_dashed_words_url)
    content = re.sub("[^0-9a-zA-Z]+", " ", content)
    tokens = word_tokenize(content)
    my_stop_words = set(stopwords.words("english")) - set(reincluded_stop_words)
    tokens = [t for t in tokens if t.lower() not in my_stop_words]

    tokens = _connect_phrases(tokens, entities_dict)

    return tokens


def _generate_regex_dashed_url(n_min_dashed_words_url):
    """
    Generate regex to parse dashed words from URL

    Parameters
    ----------
    n_min_dashed_words_url : int
        The number of words that must be separated by dashes in a URL, to treat the
        text as an actual relevant content summary

        For example, many websites have "breaking-news" in the URL, but we
        don't care to extract this as a content summary
    """
    word_and_dash = "[a-z0-9]+-"
    regexp = r"\/(" + n_min_dashed_words_url * word_and_dash + r"[a-z0-9-]*)(?:\/|\.|$)"
    return regexp


def _process_urls(content, n_min_dashed_words_url):
    """
    Remove URLs from text, but include any relevant content summaries from URL

    Parameters
    ----------
    content : str
        Text WhatsApp message
    n_min_dashed_words_url : int
        The number of words that must be separated by dashes in a URL, to treat the
        text as an actual relevant content summary

    Returns
    -------
    str
        Text WhatsApp message, with URLs replaced with any relevant content summaries

    Example
    -------
        Input:
            *Buhari Fails To Recognise Own Handwriting On NTA (Video)*
            https://www.google.com/url?sa=i&source=web&cd=&ved=2ahUKEwjAlKiMv5_gAhWVD2M
            BHRujDbQQzPwBegQIARAC&url=https%3A%2F%2Fwww.herald.ng%2Fbuhari-fails-to-rec
            ognise-own-handwriting-on-nta-video%2F&psig=AOvVaw3zF9dZ2j_HKcdSie7KkTeu&us
            t=1549280861065125
        Output:
            *Buhari Fails To Recognise Own Handwriting On NTA (Video)* buhari-fails-to-
            recognise-own-handwriting-on-nta-video
    """
    content = urllib.parse.unquote(content)

    urls = re.findall(r"((?:https?:\/\/|www)(?:[^ ]+))", content)

    for url in urls:
        # We only extract portions with N_WORDS_MIN or more words separated by dashes
        extract = re.findall(_generate_regex_dashed_url(n_min_dashed_words_url), url)
        extract = " ".join(extract)

        content = content.replace(url, extract)

    return content


def _triplewise(tokens):
    """
    Return iterable of lowercase triple-wise consecutive tokens
    """

    tokens = [t.lower() for t in tokens]
    a, b, c = tee(tokens, 3)

    # b one token ahead, c two tokens ahead
    next(b, None)
    next(c, None)
    next(c, None)

    return zip(a, b, c, range(len(tokens)))


def _connect_phrases(tokens, entities_dict):
    """
    Connect two and three word phrases with underscores,
    e.g. "African Union" => "African_Union"

    Parameters
    ----------
    tokens : list of str
    entities_dict : dict[tuple, str]
        Example: entities_dict[('african', 'union')] = "African_Union"

    Returns
    -------
    tokens_connected : list of str
        List of tokens, with entities connected.
    """

    # Can only look at triples with 3 or more tokens
    if len(tokens) >= 3:
        triples = _triplewise(tokens)

        for first, second, third, i in triples:
            if (first, second) in entities_dict:
                tokens[i] = entities_dict[(first, second)]
                tokens[i + 1] = None

            elif (first, second, third) in entities_dict:
                tokens[i] = entities_dict[(first, second, third)]
                tokens[i + 1] = None
                tokens[i + 2] = None

            elif (second, third) in entities_dict:
                tokens[i + 1] = entities_dict[(second, third)]
                tokens[i + 2] = None

    # Manually handle case with 2 tokens
    elif len(tokens) == 2:
        t0 = tokens[0].lower()
        t1 = tokens[1].lower()

        if (t0, t1) in entities_dict:
            tokens[0] = entities_dict[(t0, t1)]
            tokens[1] = None

    tokens = list(filter(None, tokens))
    return tokens
