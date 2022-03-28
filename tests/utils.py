import os
from pathlib import Path

from gensim.models import KeyedVectors


def load_wv_pretrained_bin(folder, filename):
    """
    Load pretrained word2vec model from either local mount or S3
    based on environment var.

    TODO: make into a pure function and take ENV as input
    """

    if os.getenv("ENV") == "topic_model":
        bucket = os.getenv("WORD2VEC_BINARY_BUCKET")
        model = KeyedVectors.load_word2vec_format(
            f"s3://{bucket}/{filename}", binary=True
        )

    else:
        full_path = Path(__file__).parent / "data" / folder / filename
        model = KeyedVectors.load_word2vec_format(
            full_path,
            binary=True,
        )

    return model
