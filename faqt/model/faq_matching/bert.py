class QuestionAnswerBERTScorer:
    """FAQ matching model based on (question, answer) relevance scoring using BERT"""

    def __init__(
        self, bert_model_name="distilbert-base-uncased", negative_sampling_rate=1.2
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
        """
        self.bert_model_name = bert_model_name
        self.negative_sampling_rate = negative_sampling_rate

    def set_contents(self, messages, contents, **kwargs):
        """
        "Fit" model with FAQ message (answers) and associated example questions by
        1. generating negative samples
        2. creating BERT inputs of question, answer pairs
        3. tokenizing
        4. training

        Each step may be implemented in a separate class method or
        module-level method.

        Parameters
        ----------
        messages : List[List[str]]
            `messages[i]` are the example messages associated with `contents[
            i]`.
        contents : List[str]
            FAQ contents
        """
        raise NotImplementedError

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
