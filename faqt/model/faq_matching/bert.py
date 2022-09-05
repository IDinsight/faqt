from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class QuestionAnswerBERTScorer:
    """FAQ matching model based on (question, answer) relevance scoring using BERT"""

    def __init__(self, model_path, batch_size=1):
        """
        Initialize.

        Parameters
        ----------
        model_path : str
            path to Huggingface transformers model directory
        batch_size : int, default=1
            batch size for predictions
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        classifier = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.model = pipeline(
            task="text-classification",
            model=classifier,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
        self.batch_size = batch_size

        self.messages = None
        self.contents = None
        self.num_contents = None

    @property
    def contents_set(self):
        """checks if contents are set"""
        return not (self.messages is None or self.contents is None)

    def set_contents(self, messages, contents):
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
        self.messages = list(messages)
        self.contents = list(contents)
        self.num_contents = len(contents)

        return self

    def score_contents(self, message):
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
        if not self.contents_set:
            raise ValueError(
                "Contents unavailable. Set contents first using "
                "`.set_contents(...)`."
            )
        inputs = [{"text": message, "text_pair": content} for content in self.contents]
        outputs_generator = self.model(inputs)

        relevance_scores = []

        for i, prediction in enumerate(outputs_generator):
            is_one = int(prediction["label"] == "LABEL_1")
            score = prediction["score"]

            relevance_score = (1 - is_one) * (1 - score) + is_one * score
            relevance_scores.append(relevance_score)

        return relevance_scores
