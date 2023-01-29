from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class KeywordRule:
    """Dataclass for keyword rule

    Parameters
    ------
    include : List[str]
        List of keywords that must be present. Must be pre-processed the same
        way as messages.
    exclude : List[str]
        List of keywords that must not be present. Must be pre-processed the
        same way as messages.
    """

    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Check that one of `include` and `exclude` has nonempty inputs."""
        if len(self.include) == 0 and len(self.exclude) == 0:
            raise ValueError(
                "Must provide nonempty list for at least one of `include` and "
                "`exclude`."
            )

        intersection = set(self.include).intersection(set(self.exclude))
        if len(intersection) > 0:
            raise Warning(
                f"Include and Exclude both contain the following: "
                f"{list(intersection)}. This rule will ALWAYS evaluate to "
                f"False."
            )


def evaluate_keyword_rule(message, rule):
    """
    Check if keyword rule evaluates True on tokens. Must include all
    keywords in `rule.include` and must exclude all keywords in
    `rule.exclude`

    Parameters
    ----------
    message : str or List[str]
        A string or a list of pre-processed tokens to evaluate keyword
        rules on.
    rule: KeywordRule
        A `model.keyword_rule_matching_model.KeywordRule` object.

    Returns
    -------
    bool

    """
    contains_all_includes = all(include in message for include in rule.include)
    contains_no_excludes = all(exclude not in message for exclude in rule.exclude)

    return contains_all_includes and contains_no_excludes


class UrgencyDetectionBase(ABC):
    """Base class for Urgency detection  models. Whether ML based or not."""

    def __init__(self, model, preprocessor):
        """
        Setting model (whether it is rule based or not)

        Parameters
        -----------
        model : sklearn.pipeline.Pipeline or List[faqt.model.urgency_detection.KeywordRule]
            Model to use for predictions.
        preprocessor : function
            Function to preprocess the message
        """
        self.preprocessor = preprocessor
        self.model = model

    @abstractmethod
    def predict(self, messages):
        """make prediction on the text"""
        raise NotImplementedError


class RuleBasedUD(UrgencyDetectionBase):
    """Rule-based  model"""

    def __init__(self, model, preprocessor):
        """
        Parameters
        -----------
        model : List[faqt.model.urgency_detection.KeywordRule]
            List of KeywordRule objects to use for predictions.
        preprocessor : function
            Function to preprocess the message
        """
        super(RuleBasedUD, self).__init__(model, preprocessor)

    def is_set(self):
        """Checks if rules are set"""
        return self.model is not None and len(self.model) > 0

    def predict(self, message):
        """
        Return final urgency score

        Parameters
        ----------
        message : str
            A string or a list of pre-processed tokens to evaluate keyword
            rules on.

        Returns
        -------
        float : urgency_score


        """

        scores = self.predict_scores(message)

        urgency_score = float(any(scores))

        return urgency_score

    def predict_scores(self, message):
        """
        Get urgency score for each keyworld rule

        Parameters
        ----------
        message : str or List[str]
            A string or a list of pre-processed tokens to evaluate keyword
            rules on.

        Returns
        -------
        List[float]: Urgency score for each rule in rules list


        """
        if not self.is_set():
            raise ValueError("Rules have not been added")

        preprocessed_message = self.preprocessor(message)
        evaluations = [
            evaluate_keyword_rule(preprocessed_message, rule) for rule in self.model
        ]
        scores = list(map(float, evaluations))

        return scores


class MLBasedUD(UrgencyDetectionBase):
    """Machine Learning  based  model"""

    def __init__(self, model, preprocessor):
        """
        Setting model (ML based models)

        Parameters
        -----------
        model : sklearn.models.Pipeline
            Machine Learning model to use for predictions.
        preprocessor : function
            Function to preprocess the message. During prediction, the raw text will be preprocessed using this function, and then passed to the `model`'s predict function.
        """
        super(MLBasedUD, self).__init__(model, preprocessor)

    def predict(self, message):
        """
        Return final urgency score.

        Parameters
        ----------
        message : str
            A string or a list of pre-processed tokens to classify as urgent or not.

        Returns
        -------
        float: urgency_score

        """

        preprocessed_message = " ".join(self.preprocessor(message))
        prediction = self.model.predict([preprocessed_message])
        return float(prediction)

    def get_model(self):
        """
        Return prediction model

        Returns
        -------
        sklearn.models.Pipeline: model
        """
        return self.model
