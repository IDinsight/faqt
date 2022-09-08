import os
from pathlib import Path

import pytest
from hunspell import Hunspell
from tests.utils import download_huggingface_model_from_s3, load_wv_pretrained_bin


@pytest.fixture(scope="session")
def w2v_model():
    print("Start loading Google News model.")
    return load_wv_pretrained_bin(
        "pretrained_wv_models", "GoogleNews-vectors-negative300-prenorm.bin"
    )


@pytest.fixture(scope="session")
def hunspell():
    hunspell = Hunspell()
    return hunspell


@pytest.fixture(scope="session")
def bert_model_path():
    folder = "sequence_classification_models"
    model_folder = "huggingface_model"

    if os.getenv("GITHUB_ACTIONS") == "true":
        bucket = os.getenv("WORD2VEC_BINARY_BUCKET")
        full_path = download_huggingface_model_from_s3(bucket, folder, model_folder)
    else:
        full_path = Path(__file__).parent / "data" / folder / model_folder

    return full_path
