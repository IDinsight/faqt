from dataclasses import dataclass, field
from typing import List


@dataclass
class KeywordRule:
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Check that one of `include` and `exclude` has nonempty inputs."""
        if len(self.include) == 0 and len(self.exclude) == 0:
            raise ValueError(
                "Must provide nonempty list for at least one of `include` and "
                "`exclude`."
            )


def evaluate_keyword_rule(message, rule):
    """
    Check if keyword rule evaluates True on message. Must include all
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
    contains_all_includes = all(
        include in message for include in rule.include)
    contains_no_excludes = all(
        exclude not in message for exclude in rule.exclude)

    return contains_all_includes and contains_no_excludes


def evaluate_keyword_rules(message, keyword_rules):
    """
    Evaluate keyword rules on the message.

    Parameters
    ----------
    message : str or List[str]
    A string or a list of pre-processed tokens to evaluate keyword
        rules on.
    keyword_rules: List[KeywordRule]
        List of KeywordRule objects with attributes 'include'
        and 'exclude' defining list of keywords to include/exclude. If
        the contents are dictionaries, they are converted to keyword rules.

    Returns
    -------
    evaluations : List[bool]
        List of booleans of length `len(rules)`. `evaluations[i]` is
        the evaluation of `rules[i]` on `message`
    """
    evaluated_rules = [
        evaluate_keyword_rule(message, rule)
        for rule in keyword_rules
    ]

    return evaluated_rules
