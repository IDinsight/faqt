import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from warnings import warn


class Contextualisation:
    """Contextualisation class to use the context information to calculate weights"""

    def __init__(self, faqs, contexts):
        """
        Allows the creation of the Contextualisation object and  creates the context matrix and distance_matrix

        Parameters
        ----------
        faqs: List[dict]
            List of faqs

        contexts :List[]str
            list of contexts (in order of appearance if applicable)


        """

        if len(faqs) < 1:
            warn("No faqs detected, No weight will be calculated  ")

        self.contexts = contexts
        self.binarizer = MultiLabelBinarizer(classes=self.contexts)
        self._context_matrix = self._get_context_matrix(faqs)
        self._distance_matrix = self._get_distance_matrix(self._context_matrix)

        self.b = 0.1

    def _get_context_matrix(self, faqs):
        """Get context matrix from faqs"""
        faq_contexts = [faq["context"] for faq in faqs]
        return self.binarizer.fit_transform(faq_contexts)

    def _get_distance_matrix(self, context_matrix):
        """Get context matrix from context matrix"""
        size = context_matrix.shape[1]

        a = np.abs(np.arange(-size, size))
        distance_matrix = np.empty((size, size))

        for i in np.arange(size):
            distance_matrix[i] = a[size - i : size - i + size]

        return distance_matrix

    def _inbound_content_vector(self, inbound_content):
        """Get get inbound content as vector"""
        return [
            self.contexts.index(content)
            for content in inbound_content
            if content in self.contexts
        ]

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

        if len(inbound_vector) < 1:
            raise ValueError(" Inbound content cannot be empty")
        D = self._distance_matrix[inbound_vector].min(axis=0)

        rbf_weights = rbf(self.b, D)
        weights = (rbf_weights * self._context_matrix).max(axis=1)
        return weights
