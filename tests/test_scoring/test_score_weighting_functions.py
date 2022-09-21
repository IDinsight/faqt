import numpy as np
import pytest
from faqt.scoring.score_weighting import add_weight


@pytest.mark.parametrize(
    "scores, weights, N, expected_output",
    [
        (0, 1, 1, 0.5),
        ([0], [1], 0, np.asarray([0.0])),
        ([0], [1], 1, np.asarray([0.5])),
        ([1, 1], [1, 0], 1, np.asarray([1.0, 0.5])),
        ([1, 1], [0, 0], 1, np.asarray([0.5, 0.5])),
        ([1, 1], [0, 0], 0, np.asarray([1.0, 1.0])),
    ],
)
def test_add_weight(scores, weights, N, expected_output):
    output = add_weight(scores, weights, N=N)

    if isinstance(output, np.ndarray):
        assert np.array_equal(output, expected_output)
    else:
        assert output == expected_output
