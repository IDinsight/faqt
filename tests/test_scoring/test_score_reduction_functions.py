import pytest
from faqt.scoring.score_reduction import avg_min_mean, simple_mean


@pytest.mark.parametrize(
    "values, expected_output",
    [
        (
            [
                0.0,
            ],
            0.0,
        ),
        (
            [
                1.0,
            ],
            1.0,
        ),
        (
            [
                0.0,
                0.0,
                0.0,
            ],
            0.0,
        ),
        ([1.0, 1.0, 1.0], 1.0),
        (
            [
                0.0,
                2.0,
            ],
            0.5,
        ),
        ([0.0, 1.0, 5.0], 1.0),
    ],
)
def test_avg_min_mean(values, expected_output):
    output = avg_min_mean(values)

    assert output == expected_output


@pytest.mark.parametrize(
    "values, expected_output",
    [
        (
            [
                0.0,
            ],
            0.0,
        ),
        (
            [
                1.0,
            ],
            1.0,
        ),
        (
            [
                0.0,
                0.0,
                0.0,
            ],
            0.0,
        ),
        ([1.0, 1.0, 1.0], 1.0),
        (
            [
                0.0,
                1.0,
            ],
            0.5,
        ),
        ([0.0, 1.0, 5.0], 2.0),
    ],
)
def test_simple_mean(values, expected_output):
    output = simple_mean(values)

    assert output == expected_output
