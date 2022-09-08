from pathlib import Path

import numpy as np
import pytest
import yaml
from faqt.model import StepwiseKeyedVectorScorer
from faqt.preprocessing import preprocess_text_for_word_embedding

pytestmark = pytest.mark.slow


class TestStepwiseKeyedVectorScorer:
    """Create a model with just the bare bones"""

    sample_messages = [
        "\U0001f600\U0001f929\U0001f617",
        """ I'm worried about the vaccines. Can I have some information? \U0001f600
            πλέων ἐπὶ οἴνοπα πόντον ἐπ᾽ ἀλλοθρόους ἀνθρώπους, ἐς Τεμέσην
        """,
        "cuoisn mircochippde my ddady with vcacines",
        "",
        """ The sun sets,
            In a silent sky,
            Echoing of a passing song,
            A long wait for the dawn,
            A record of time while it glides along.
        """,
    ]

    @pytest.fixture
    def basic_model(self, w2v_model):
        faqt = StepwiseKeyedVectorScorer(
            w2v_model,
            tag_scoring_method="cs_nearest_k_percent_average",
            tag_scoring_kwargs=dict(k=100, floor=1),
            score_reduction_method="simple_mean",
            score_reduction_kwargs=None,
        )
        return faqt

    @pytest.fixture
    def weighted_scoring_model(self, w2v_model):
        faqt = StepwiseKeyedVectorScorer(
            w2v_model,
            tag_scoring_method="cs_nearest_k_percent_average",
            tag_scoring_kwargs=dict(k=100, floor=1),
            score_reduction_method="simple_mean",
            score_reduction_kwargs=None,
            weighting_method="add_weight",
            weighting_kwargs={"N": 1.0},
        )
        return faqt

    @pytest.fixture
    def hunspell_model(self, w2v_model, hunspell):
        faqt = StepwiseKeyedVectorScorer(
            w2v_model,
            tag_scoring_method="cs_nearest_k_percent_average",
            tag_scoring_kwargs=dict(k=100, floor=1),
            score_reduction_method="simple_mean",
            score_reduction_kwargs=None,
            hunspell=hunspell,
            tags_guiding_typos=["music", "food"],
        )
        return faqt

    @pytest.fixture(scope="class")
    def tagsets(self):
        full_path = Path(__file__).parent / "data/tag_test_data.yaml"
        with open(full_path) as file:
            yaml_dict = yaml.full_load(file)

        tagsets_data = yaml_dict["tags_refresh_data"]
        return [tagset_data["tags"] for tagset_data in tagsets_data]

    @pytest.fixture(scope="class")
    def tagset_weights(self):
        full_path = Path(__file__).parent / "data/tag_test_data.yaml"
        with open(full_path) as file:
            yaml_dict = yaml.full_load(file)

        tagsets_data = yaml_dict["tags_refresh_data"]
        return [tagset_data["weight"] for tagset_data in tagsets_data]

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_setting_contents_without_weights_sets_all_tag_attr(
        self,
        basic_model,
        tagsets,
    ):
        basic_model.set_contents(tagsets, weights=None)

        assert hasattr(basic_model, "tagsets")
        assert hasattr(basic_model, "tagset_wvs")
        assert hasattr(basic_model, "tagset_weights")

    @pytest.mark.parametrize("input_text", sample_messages)
    def test_basic_model_score_with_empty_tags_returns_empty_scores(
        self, basic_model, input_text
    ):

        tokens = preprocess_text_for_word_embedding(input_text, {}, 0)
        basic_model.set_contents([])
        assert len(basic_model.tagsets) == 0

        result = basic_model.score_contents(
            tokens, return_tag_scores=False, return_spell_corrected=False
        )
        scores = result["overall_scores"]
        assert len(scores) == 0

    @pytest.mark.parametrize(
        "input_text",
        sample_messages,
    )
    def test_basic_model_score_with_nonempty_tags_returns_nonempty_scores(
        self, hunspell_model, tagsets, input_text
    ):
        hunspell_model.set_contents(tagsets)
        assert len(hunspell_model.tagsets) == len(tagsets)
        tokens = preprocess_text_for_word_embedding(input_text, {}, 0)

        result = hunspell_model.score_contents(
            tokens, return_tag_scores=False, return_spell_corrected=False
        )
        scores = result["overall_scores"]

        if len(tokens) == 0:
            assert len(scores) == 0
        else:
            assert len(scores) == len(tagsets)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        "input_text",
        sample_messages,
    )
    def test_basic_model_with_weights_returns_weighted_scores(
        self, basic_model, weighted_scoring_model, tagsets, tagset_weights, input_text
    ):
        tokens = preprocess_text_for_word_embedding(input_text, {}, 0)
        weights = np.asarray(tagset_weights) / np.sum(tagset_weights)

        basic_model.set_contents(tagsets, weights=None)
        unweighted_result = basic_model.score_contents(tokens)
        unweighted_scores = unweighted_result["overall_scores"]

        weighted_scoring_model.set_contents(tagsets, weights=tagset_weights)
        weighted_result = weighted_scoring_model.score_contents(tokens)
        weighted_scores = weighted_result["overall_scores"]

        # Expects N=1.0
        for i, (u, w) in enumerate(zip(unweighted_scores, weighted_scores)):
            assert w == (u + weights[i]) / 2

    def test_weights_correctly_calculated_with_weights(
        self, weighted_scoring_model, tagsets, tagset_weights
    ):
        weighted_scoring_model.set_contents(tagsets, weights=tagset_weights)

        assert np.isclose(sum(weighted_scoring_model.tagset_weights), 1.0)
        assert np.allclose(
            np.asarray(weighted_scoring_model.tagset_weights),
            np.asarray(tagset_weights) / sum(tagset_weights),
        )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_scores_with_weights_increase_rank(
        self,
        basic_model,
        weighted_scoring_model,
        tagsets,
        tagset_weights,
    ):
        tokens = preprocess_text_for_word_embedding(
            "I love the outdoors. What should I pack for lunch?", {}, 0
        )

        basic_model.set_contents(tagsets, weights=tagset_weights)
        unweighted_result = basic_model.score_contents(tokens)
        unweighted_scores = unweighted_result["overall_scores"]
        ranks_mean_plus_weight = np.argsort(unweighted_scores)

        weighted_scoring_model.set_contents(tagsets, weights=None)
        weighted_result = weighted_scoring_model.score_contents(tokens)
        weighted_scores = weighted_result["overall_scores"]
        ranks_simple_mean = np.argsort(weighted_scores)

        assert np.argwhere(ranks_simple_mean == 1) <= np.argwhere(
            ranks_mean_plus_weight == 1
        )
        assert np.argwhere(ranks_simple_mean == 2) <= np.argwhere(
            ranks_mean_plus_weight == 2
        )

    def test_scoring_without_setting_raises_error(self, basic_model):
        with pytest.raises(
            ValueError, match="Set contents with `self\.set_contents\(\)`"
        ):
            basic_model.score_contents("test message")
