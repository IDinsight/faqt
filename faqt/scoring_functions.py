"""Module for various scoring functions"""

import numpy as np

__all__ = ["cs_nearest_k_percent_average"]


def cs_nearest_k_percent_average(list_of_wvs, target_wv, k=100, floor=1):
    """
    Returns the cosine similarity between target_wv and
    the average of the normalized k% of word vectors in
    list_of_wvs that are most similar to target_wv

    Parameters
    ----------
    list_of_wvs : list of vectors (1-D arrays)
    target_wv : vector (1-D array)
    k : float
        Nearest % of word vectors to capture
    floor : int
        Minimum # of word vectors to capture

    Returns
    -------
    float
        Cosine similarity between target_wv and
        the mean of the k% of word vectors (or floor, whichever is larger)
        in list_of_wvs that have highest CS to target_wv

    Notes
    -----
    All vectors are expected to be already normalized
    """
    avg = _nearest_k_percent_average(list_of_wvs, target_wv, k, floor)
    avg_dot = np.dot(avg, avg)

    if np.isclose(avg_dot, 0):
        return 0
    else:
        return np.dot(avg, target_wv) / np.sqrt(avg_dot)


def _nearest_k_percent(list_of_wvs, target_wv, k, floor):
    """
    Returns an array of shape (n_vectors, n_dim_embeddings)
    which contains the k% of word vectors (or floor, whichever is larger)
    in list_of_wvs that have highest CS to target_wv

    Parameters
    ----------
    list_of_wvs : list of vectors (1-D arrays)
    target_wv : vector (1-D array)
    k : float
        Nearest % of word vectors to capture
    floor : int
        Minimum # of word vectors to capture

    Returns
    -------
    ndarray
        (n_vectors, n_dim_embeddings)
        which contains the k% of word vectors (or floor, whichever is larger)
        in list_of_wvs that have highest CS to target_wv
    """
    cs = [np.dot(wv, target_wv) for wv in list_of_wvs]

    # k%, or 3, whichever is greater
    n_top = np.max([floor, int(k / 100 * len(list_of_wvs))])
    # No more than the entirety of wvs!
    n_top = np.min([n_top, len(list_of_wvs)])

    indices = np.argsort(cs)[::-1][:n_top]

    return np.vstack(list_of_wvs)[indices]


def _nearest_k_percent_average(list_of_wvs, target_wv, k, floor):
    """
    Obtains the k% of word vectors in list_of_wvs
    that are most similar to target_wv, then takes the mean

    Parameters
    ----------
    list_of_wvs : list of vectors (1-D arrays)
    target_wv : vector (1-D array)
    k : float
        Nearest % of word vectors to capture
    floor : int
        Minimum # of word vectors to capture

    Returns
    -------
    vector (1-D array)
        Mean of k% of word vectors (or floor, whichever is larger)
        in list_of_wvs that have highest CS to target_wv

    Notes
    -----
    All vectors are expected to be already normalized
    """
    nearest_wvs = _nearest_k_percent(list_of_wvs, target_wv, k, floor)

    return np.mean(nearest_wvs, axis=0)
