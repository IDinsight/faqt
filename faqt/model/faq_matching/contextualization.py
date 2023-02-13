import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from warnings import warn

b = 0.1


class Contextualization:
    """

    Contextualization class to use  context information to calculate weights.

    Contextualization can be used to calculate weights to be attributed to each content while scoring.
    This weight is calculated using some contexts obtained from each content and the context of the message.


    Parameters
    ----------
    contents_contexts: List[List[str]]
        List of contents contexts.
    contents_id: List[str]
        List of contents id

    distance_matrix :pandas.DataFrame
        A square matrix in the form of a pandas dataframe with the contexts list as
        both columns and index and distance between each pair of contexts as values.


    """

    def __init__(self, contents_id, contents_context, distance_matrix):
        """Define constructor"""
        if len(contents_context) < 1:
            warn("No faqs detected, No weight will be calculated.")

        if len(distance_matrix) < 1:
            raise ValueError(
                "Empty dataframe, please provided a distance matrix as a dataframe"
            )
        self.contexts = list(distance_matrix.columns)
        self.contents_id = contents_id
        self.binarizer = MultiLabelBinarizer(classes=self.contexts)
        self._context_matrix = self._get_context_matrix(contents_context)
        self._distance_matrix = distance_matrix.values

        self.b = b

    def _get_context_matrix(self, content_contexts):
        """Get context matrix from contents"""
        return self.binarizer.fit_transform(content_contexts)

    def _message_context_vector(self, message_context):
        """Get message content as vector"""

        if len(message_context) < 1:
            raise ValueError("Inbound content cannot be empty")

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

        def rbf(b, d):
            return np.exp(-((b * d) ** 2))

        message_vector = self._message_context_vector(message_context)

        D = self._distance_matrix[message_vector].min(axis=0)

        rbf_weights = rbf(self.b, D)
        weights = (rbf_weights * self._context_matrix).max(axis=1)
        content_weights = {
            content_id: weight
            for (content_id, weight) in zip(self.contents_id, weights)
        }
        return content_weights


def get_ordered_distance_matrix(context_list):
    """Get context matrix from context list"""
    size = len(context_list)

    a = np.abs(np.arange(-size, size))
    distance_matrix = np.empty((size, size))

    for i in np.arange(size):
        distance_matrix[i] = a[size - i : size - i + size]
    distance_matrix = pd.DataFrame(
        distance_matrix, columns=context_list, index=context_list, dtype=int
    )
    return distance_matrix
