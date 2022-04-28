import re
import urllib


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
    regexp = (
        r"\/(" + (n_min_dashed_words_url - 1) * word_and_dash + r"["
        r"a-z0-9-]*)(?:\/|\.|$)"
    )
    return regexp


def process_urls(content, n_min_dashed_words_url):
    """
    Remove URLs from text, but include any relevant content summaries from URL

    Parameters
    ----------
    content : str
        Text WhatsApp message
    n_min_dashed_words_url : int
        The number of words that must be separated by dashes in a URL, to treat the
        text as an actual relevant content summary. We only allow 2 or more
        words to be extracted

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
    if n_min_dashed_words_url <= 1:
        n_min_dashed_words_url = 2

    content = urllib.parse.unquote(content)

    urls = re.findall(r"((?:https?:\/\/|www)(?:[^ ]+))", content)

    for url in urls:
        # We only extract portions with N_WORDS_MIN or more words separated by dashes
        extract = re.findall(_generate_regex_dashed_url(n_min_dashed_words_url), url)
        extract = " ".join(extract)

        content = content.replace(url, extract)

    return content


def remove_punctuation(text):
    """Removes punctuations. Only alphabets and numbers are kept."""
    return re.sub("[^0-9a-zA-Z]+", " ", text)