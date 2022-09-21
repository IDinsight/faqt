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


def evaluate_keyword_rules(message, keyword_rules):
    """
    Evaluate keyword rules on the tokens.

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
        the evaluation of `rules[i]` on `tokens`
    """
    evaluated_rules = [evaluate_keyword_rule(message, rule) for rule in keyword_rules]

    return evaluated_rules
