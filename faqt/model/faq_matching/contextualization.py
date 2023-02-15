import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from warnings import warn

VARIANCE = 0.1


class Contextualization:
    """
    Contextualization class to use  context information to calculate weights.

    Contextualization can be used to calculate weights to be attributed to each content while scoring.
    This weight is calculated using some contexts obtained from each content and the context of the message.

    Parameters
    ----------
    contents_contexts: Dict[str, List[str]]
        Dictionnary of contents with the contents ID as key and the contexts list as value.
    distance_matrix :pandas.DataFrame
        A square matrix in the form of a pandas dataframe with the contexts list as
        both columns and index and distance between each pair of contexts as values.
    variance: float
        The variance parameter for the kernelization using the radial basis function.

    """

    def __init__(self, contents_dict, distance_matrix, variance=VARIANCE):
        """Define constructor"""

        self.check_inputs(contents_dict, distance_matrix)
        self.contexts = list(distance_matrix.columns)
        self.contents_id = contents_dict.keys()
        self.binarizer = MultiLabelBinarizer(classes=self.contexts)
        self._context_matrix = self._get_context_matrix(list(contents_dict.values()))
        self._distance_matrix = distance_matrix.values

        self.variance = variance

    def check_inputs(self, contents_dict, distance_matrix):
        assert (
            len(distance_matrix) > 0
        ), "Empty dataframe, please provided a distance matrix as a dataframe"
        assert (
            len(distance_matrix.shape) == 2
            and distance_matrix.shape[0] == distance_matrix.shape[1]
        ), "Distance matrix is not a square matrix"
        if len(contents_dict) < 1:
            warn("No faqs detected, No weight will be calculated.")
        else:
            flattened_contexts = np.hstack(list(contents_dict.values()))
            unique_values = np.unique(flattened_contexts)
            invalid = np.setdiff1d(unique_values, distance_matrix.columns)
            if len(invalid) > 0:
                raise ValueError(
                    f"contexts {str(invalid)} cannot be found in 'distance_matrix'"
                )

    def _get_context_matrix(self, content_contexts):
        """Convert contexts provided as list of strings into a binary vector representation"""
        return self.binarizer.fit_transform(content_contexts)

    def _message_context_vector(self, message_context):
        """Convert message context list into vector of indexes as they appear in the content context list"""

        if len(message_context) < 1:
            raise ValueError("Message context cannot be empty")

        message_vector = [
            self.contexts.index(value)
            for value in message_context
            if value in self.contexts
        ]
        if len(message_vector) != len(message_context):
            invalid = [value for value in message_context if value not in self.contexts]
            raise ValueError(f"Unknown contexts : {str(invalid)} ")
        else:
            return message_vector

    def get_context_weights(self, message_context):
        """
        Get context weights from the message contexts.

        Parameters
        ----------


        message_context :List[str]
            list of contexts

        Returns
        -------
        weights : list of str
            List of tokens, with entities connected.
        """

        def rbf(variance, vectors):
            return np.exp(-((variance * vectors) ** 2))

        message_vector = self._message_context_vector(message_context)

        distance_vectors = self._distance_matrix[message_vector].min(axis=0)

        rbf_weights = rbf(self.variance, distance_vectors)
        weights = (rbf_weights * self._context_matrix).max(axis=1)
        content_weights = {
            content_id: weight
            for (content_id, weight) in zip(self.contents_id, weights)
        }
        return content_weights


def get_ordered_distance_matrix(context_list):
    """Create a distance matrix by asssuming that the distance between each adjacent context is 1"""
    size = len(context_list)

    a = np.abs(np.arange(-size, size))
    distance_matrix = np.empty((size, size))

    for i in np.arange(size):
        distance_matrix[i] = a[size - i : 2 * size - i]
    distance_matrix = pd.DataFrame(
        distance_matrix, columns=context_list, index=context_list, dtype=int
    )
    return distance_matrix
