import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from warnings import warn


class Contextualisation:
    """Contextualisation class to use the context information to calculate weights"""

    def __init__(self, faqs, distance_matrix):
        """
        Allows the creation of the Contextualisation object and  creates the context matrix and distance_matrix

        Parameters
        ----------
        faqs: List[dict]
            List of faqs

        distance_matrix :pandas.DataFrame
            A matrix as form of a pandas dataframe with the contexts list as columns and index. and distance between each context as value.


        """

        if len(faqs) < 1:
            warn("No faqs detected, No weight will be calculated.")

        if len(distance_matrix) < 1:
            raise ValueError(
                "Empty dataframe, please provided a distance matrix as a dataframe"
            )
        self.contexts = list(distance_matrix.columns)
        self.binarizer = MultiLabelBinarizer(classes=self.contexts)
        self._context_matrix = self._get_context_matrix(faqs)
        self._distance_matrix = distance_matrix.values

        self.b = 0.1

    def _get_context_matrix(self, faqs):
        """Get context matrix from faqs"""
        faq_contexts = [faq["context"] for faq in faqs]
        return self.binarizer.fit_transform(faq_contexts)

    def _inbound_content_vector(self, inbound_content):
        """Get get inbound content as vector"""

        if len(inbound_content) < 1:
            raise ValueError("Inbound content cannot be empty")

        inbound_vector = [
            self.contexts.index(value)
            for value in inbound_content
            if value in self.contexts
        ]
        if len(inbound_vector) != len(inbound_content):
            invalid = [value for value in inbound_content if value not in self.contexts]
            raise ValueError(f"Unknown contexts : {str(invalid)} ")
        else:
            return inbound_vector

    def get_context_weights(self, inbound_content):
        """
        Get context weights from content.

        Parameters
        ----------


        contexts :List[str]
            list of content


        """

        def rbf(b, d):
            return np.exp(-((b * d) ** 2))

        inbound_vector = self._inbound_content_vector(inbound_content)

        D = self._distance_matrix[inbound_vector].min(axis=0)

        rbf_weights = rbf(self.b, D)
        weights = (rbf_weights * self._context_matrix).max(axis=1)
        return weights


def get_ordered_distance_matrix(context_list):
    """Get context matrix from context lis"""
    size = len(context_list)

    a = np.abs(np.arange(-size, size))
    distance_matrix = np.empty((size, size))

    for i in np.arange(size):
        distance_matrix[i] = a[size - i : size - i + size]
    distance_matrix = pd.DataFrame(
        distance_matrix, columns=context_list, index=context_list, dtype=int
    )
    return distance_matrix
