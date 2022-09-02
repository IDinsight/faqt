from itertools import chain

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class QuestionAnswerBERTScorer:
    """FAQ matching model based on (question, answer) relevance scoring using BERT"""

    def __init__(
        self,
        bert_model_name="distilbert-base-uncased",
        negative_sampling_rate=1.2,
        random_state=None,
    ):
        """

        Parameters
        ----------
        bert_model_name : str
            BERT model ID in huggingface. See
            [transformers.AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/v4.21.1/en/model_doc/auto#transformers.AutoModelForSequenceClassification).
        negative_sampling_rate : float
            Desired ratio <negative samples> / <positive samples>. Default is
            1.2.
        random_state : int, optional
        """
        self.bert_model_name = bert_model_name
        self.negative_sampling_rate = negative_sampling_rate
        self.random_state = random_state  # TODO: save/set random state

    def set_contents(self, messages, contents, **kwargs):
        """
        "Fit" model with FAQ content (answers) and associated example questions by
        1. generating negative samples
        2. creating BERT inputs of question, answer pairs
        3. tokenizing
        4. training

        Each step may be implemented in a separate class method or
        module-level method.

        Parameters
        ----------
        messages : List-like[List[str]]
            `messages[i]` are the example messages associated with `contents[
            i]`.
        contents : List-like[str]
            FAQ contents.
        """
        messages = list(messages)
        contents = list(contents)

        # Generate negative samples
        negative_samples = {
            "messages": [],
            "contents": [],
        }
        n_negative = 0

        for i, (correct_messages, content) in enumerate(zip(messages, contents)):
            wrong_msgs = list(chain(messages[: i - 1] + messages[i + 1 :]))
            n = np.around(len(correct_messages) * self.negative_sampling_rate)
            sampled_wrong_msgs = np.random.choice(wrong_msgs, size=n, replace=False)

            negative_samples["messages"].extend(sampled_wrong_msgs)
            negative_samples["contents"].extend([content] * n)
            n_negative += n

        messages_aug = messages + negative_samples["messages"]
        contents_aug = contents + negative_samples["contents"]

        n_positive = len(contents)
        labels = np.ones(n_positive) + np.zeros(n_negative)

    def score_contents(self, message, **kwargs):
        """
        Score message against each of the contents

        Parameters
        ----------
        message : str
        kwargs :

        Returns
        -------
        tag_scores

        """
        pass
