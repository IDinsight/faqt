__version__ = "1.0.2"

from .model import (
    KeywordRule,
    QuestionAnswerBERTScorer,
    StepwiseKeyedVectorsScorer,
    WMDScorer,
)
from .preprocessing import (
    preprocess_text_for_keyword_rule,
    preprocess_text_for_word_embedding,
)
