import types

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class QuestionAnswerBERTScorer:
    """FAQ matching model based on (question, answer) relevance scoring using BERT"""

    def __init__(self, model_path, batch_size=1):
        """
        Initialize.

        Parameters
        ----------
        model_path : str
            Path to HuggingFace transformers model directory. If trained
            locally, it should be the directory passed to
            `transformers.Trainer.save_model()` function
            or `transformers.Model.save_pretrained()` function.
        batch_size : int, default=1
            Batch size for predictions
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        classifier = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.model = self._create_safe_pipeline(tokenizer, classifier, batch_size)

        self.batch_size = batch_size

        self.contents = None

    @staticmethod
    def _create_safe_pipeline(tokenizer, classifier, batch_size):
        """Workaround to ensure truncation during tokenization.
        See https://stackoverflow.com/a/71243383/7664921, but we adapt the
        preprocess function for text_classification task"""
        pipe = pipeline(
            task="text-classification",
            model=classifier,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

        MAX_LENGTH = classifier.config.max_position_embeddings

        def _preprocess_with_truncation(self, inputs, **tokenizer_kwargs):
            """Preprocess function to override default. Adds truncation
            parameters"""
            return_tensors = self.framework
            tokenizer_kwargs.update(
                {"truncation": "longest_first", "max_length": MAX_LENGTH}
            )

            return self.tokenizer(
                **inputs, return_tensors=return_tensors, **tokenizer_kwargs
            )

        pipe.preprocess = types.MethodType(_preprocess_with_truncation, pipe)

        return pipe

    @property
    def is_set(self):
        """Check if contents are set"""
        return self.contents is not None

    def set_contents(self, contents):
        """
        Saves contents for this scorer

        Parameters
        ----------
        contents : List-like[str]
            FAQ contents.
        """
        self.contents = list(contents)

        return self

    def score_contents(self, message):
        """
        Score message against each of the contents

        Parameters
        ----------
        message : str
            Message to score each content against

        Returns
        -------
        scores : list[float]
            List of scores for each content, in the order saved in
            `self.contents`

        """
        if not self.is_set:
            raise ValueError(
                "Contents unavailable. Set contents first using "
                "`self.set_contents()`."
            )

        inputs = [{"text": message, "text_pair": content} for content in self.contents]
        outputs_generator = self.model(inputs)

        scores = []

        for i, prediction in enumerate(outputs_generator):
            is_one = int(prediction["label"] == "LABEL_1")
            _score = prediction["score"]

            score = (1 - is_one) * (1 - _score) + is_one * _score
            scores.append(score)

        return_dict = {"overall_scores": scores}
        return return_dict
