import pytest
from tests.utils import load_wv_pretrained_bin
from hunspell import Hunspell


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
