from functools import partial
from pathlib import Path

import numpy as np
import pytest
import yaml
from faqt.model import WMDScorer
from faqt.preprocessing import preprocess_text_for_word_embedding

pytestmark = pytest.mark.slow


class TestWMDScorer:
    """Create a model with just the bare bones"""

    sample_messages = [
        "\U0001f600\U0001f929\U0001f617",
        """ I'm worried 
    about the vaccines. Can I have some information? \U0001f600
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
        faqt = WMDScorer(
            w2v_model,
            tokenizer=partial(
                preprocess_text_for_word_embedding,
                entities_dict={},
                n_min_dashed_words_url=0,
            ),
        )
        return faqt

    @pytest.fixture
    def weighted_scoring_model(self, w2v_model):
        faqt = WMDScorer(
            w2v_model,
            tokenizer=partial(
                preprocess_text_for_word_embedding,
                entities_dict={},
                n_min_dashed_words_url=0,
            ),
            weighting_method="add_weight",
            weighting_kwargs={"N": 1.0},
        )
        return faqt

    @pytest.fixture
    def hunspell_model(self, w2v_model, hunspell):
        faqt = WMDScorer(
            w2v_model,
            tokenizer=partial(
                preprocess_text_for_word_embedding,
                entities_dict={},
                n_min_dashed_words_url=0,
            ),
            hunspell=hunspell,
            tags_guiding_typos=["music", "food"],
        )
        return faqt

    @pytest.fixture(scope="class")
    def contents(self):
        full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
        with open(full_path) as file:
            yaml_dict = yaml.full_load(file)

        tagsets_data = yaml_dict["tags_refresh_data"]
        return [tagset_data["content_to_send"] for tagset_data in tagsets_data]

    @pytest.fixture(scope="class")
    def tagsets(self):
        full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
        with open(full_path) as file:
            yaml_dict = yaml.full_load(file)

        tagsets_data = yaml_dict["tags_refresh_data"]
        return [tagset_data["tags"] for tagset_data in tagsets_data]

    @pytest.fixture(scope="class")
    def content_weights(self):
        full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
        with open(full_path) as file:
            yaml_dict = yaml.full_load(file)

        tagsets_data = yaml_dict["tags_refresh_data"]
        return [tagset_data["weight"] for tagset_data in tagsets_data]

    def test_scoring_without_setting_raises_error(self, basic_model):
        with pytest.raises(
            ValueError, match="Set contents with `self\.set_contents\(\)`"
        ):
            basic_model.score_contents("test message")

    def test_setting_contents_with_list_of_list_of_str(self, basic_model, tagsets):
        basic_model.set_contents(tagsets)
        assert basic_model.contents is not None

    def test_setting_contents_with_list_of_str(self, basic_model, contents):
        basic_model.set_contents(contents)
        assert basic_model.contents is not None

    def test_setting_contents_with_mixed_content_types_raises_error(
        self, basic_model, tagsets
    ):
        mixed_type_contents = tagsets.copy()
        mixed_type_contents[1] = " ".join(tagsets[1])
        mixed_type_contents[2] = " ".join(tagsets[2])

        with pytest.raises(TypeError):
            basic_model.set_contents(mixed_type_contents)

    @pytest.mark.parametrize(
        "input_text",
        sample_messages,
    )
    def test_return_spell_corrected_flag(self, basic_model, tagsets, input_text):
        basic_model.set_contents(tagsets)
        result = basic_model.score_contents(input_text, return_spell_corrected=True)

        assert "spell_corrected" in result

    def test_resetting_contents_without_weights_is_allowed_with_warning(
        self, weighted_scoring_model, tagsets, content_weights
    ):
        weighted_scoring_model.set_contents(tagsets, weights=content_weights)
        with pytest.raises(UserWarning):
            weighted_scoring_model.set_contents([], weights=None)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_setting_contents_without_weights_sets_all_tag_attr(
        self,
        basic_model,
        tagsets,
    ):
        basic_model.set_contents(tagsets, weights=None)

        assert basic_model.contents is not None
        assert basic_model.content_vectors is not None
        assert basic_model.content_weights is None

    @pytest.mark.parametrize("input_text", sample_messages)
    def test_basic_model_score_with_empty_tags_returns_empty_scores(
        self, basic_model, input_text
    ):

        basic_model.set_contents([])
        assert len(basic_model.contents) == 0

        result = basic_model.score_contents(
            input_text, return_tag_scores=False, return_spell_corrected=False
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
        assert len(hunspell_model.contents) == len(tagsets)

        result = hunspell_model.score_contents(
            input_text, return_tag_scores=False, return_spell_corrected=True
        )
        scores = result["overall_scores"]
        spell_corrected = result["spell_corrected"]

        if len(spell_corrected) == 0:
            assert len(scores) == 0
        else:
            assert len(scores) == len(tagsets)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        "input_text",
        sample_messages,
    )
    def test_basic_model_with_weights_returns_weighted_scores(
        self, basic_model, weighted_scoring_model, tagsets, content_weights, input_text
    ):
        weights = np.asarray(content_weights) / np.sum(content_weights)

        basic_model.set_contents(tagsets, weights=None)
        unweighted_result = basic_model.score_contents(input_text)
        unweighted_scores = unweighted_result["overall_scores"]

        weighted_scoring_model.set_contents(tagsets, weights=content_weights)
        weighted_result = weighted_scoring_model.score_contents(input_text)
        weighted_scores = weighted_result["overall_scores"]

        # Expects N=1.0
        for i, (u, w) in enumerate(zip(unweighted_scores, weighted_scores)):
            assert w == (u + weights[i]) / 2

    def test_weights_correctly_calculated_with_weights(
        self, weighted_scoring_model, tagsets, content_weights
    ):
        weighted_scoring_model.set_contents(tagsets, weights=content_weights)

        assert np.isclose(sum(weighted_scoring_model.content_weights), 1.0)
        assert np.allclose(
            np.asarray(weighted_scoring_model.content_weights),
            np.asarray(content_weights) / sum(content_weights),
        )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_scores_with_weights_increase_rank(
        self,
        basic_model,
        weighted_scoring_model,
        tagsets,
        content_weights,
    ):
        input_text = "I love the outdoors. What should I pack for lunch?"

        basic_model.set_contents(tagsets, weights=content_weights)
        unweighted_result = basic_model.score_contents(input_text)
        unweighted_scores = unweighted_result["overall_scores"]
        ranks_mean_plus_weight = np.argsort(unweighted_scores)

        weighted_scoring_model.set_contents(tagsets, weights=None)
        weighted_result = weighted_scoring_model.score_contents(input_text)
        weighted_scores = weighted_result["overall_scores"]
        ranks_simple_mean = np.argsort(weighted_scores)

        assert np.argwhere(ranks_simple_mean == 1) <= np.argwhere(
            ranks_mean_plus_weight == 1
        )
        assert np.argwhere(ranks_simple_mean == 2) <= np.argwhere(
            ranks_mean_plus_weight == 2
        )
