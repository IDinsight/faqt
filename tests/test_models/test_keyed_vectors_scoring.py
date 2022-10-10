from functools import partial
from pathlib import Path

import numpy as np
import pytest
import yaml
from faqt.model import StepwiseKeyedVectorsScorer, WMDScorer
from faqt.preprocessing import preprocess_text_for_word_embedding
from gensim.similarities.docsim import WmdSimilarity

pytestmark = pytest.mark.slow

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


@pytest.fixture(scope="module")
def contents():
    full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    tagsets_data = yaml_dict["tags_refresh_data"]
    return [tagset_data["content_to_send"] for tagset_data in tagsets_data]


@pytest.fixture(scope="module")
def tagsets():
    full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    tagsets_data = yaml_dict["tags_refresh_data"]
    return [tagset_data["tags"] for tagset_data in tagsets_data]


@pytest.fixture(scope="module")
def content_weights():
    full_path = Path(__file__).parents[1] / "data/tag_test_data.yaml"
    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    tagsets_data = yaml_dict["tags_refresh_data"]
    return [tagset_data["weight"] for tagset_data in tagsets_data]


@pytest.fixture
def stepwise_basic_model(w2v_model):
    faqt = StepwiseKeyedVectorsScorer(
        w2v_model,
        tokenizer=partial(
            preprocess_text_for_word_embedding,
            entities_dict={},
            n_min_dashed_words_url=0,
        ),
        tag_scoring_method="cs_nearest_k_percent_average",
        tag_scoring_kwargs=dict(k=100, floor=1),
        score_reduction_method="simple_mean",
        score_reduction_kwargs=None,
    )
    return faqt


@pytest.fixture
def stepwise_weighted_model(w2v_model):
    faqt = StepwiseKeyedVectorsScorer(
        w2v_model,
        tokenizer=partial(
            preprocess_text_for_word_embedding,
            entities_dict={},
            n_min_dashed_words_url=0,
        ),
        tag_scoring_method="cs_nearest_k_percent_average",
        tag_scoring_kwargs=dict(k=100, floor=1),
        score_reduction_method="simple_mean",
        score_reduction_kwargs=None,
        weighting_method="add_weight",
        weighting_kwargs={"N": 1.0},
    )
    return faqt


@pytest.fixture
def wmd_basic_model(w2v_model):
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
def wmd_weighted_model(w2v_model):
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


class TestKeyedVectorsScorers:
    @pytest.mark.parametrize("model_name", ["stepwise_basic_model", "wmd_basic_model"])
    def test_scoring_without_setting_raises_error(self, model_name, request):
        basic_model = request.getfixturevalue(model_name)
        with pytest.raises(
            ValueError, match="Set contents with `self\.set_contents\(\)`"
        ):
            basic_model.score_contents("test message")

    @pytest.mark.parametrize(
        "input_text",
        sample_messages,
    )
    @pytest.mark.parametrize(
        "model_name, contents_name",
        [
            ("stepwise_basic_model", "tagsets"),
            ("wmd_basic_model", "contents"),
        ],
    )
    def test_return_spell_corrected_flag(
        self, model_name, contents_name, input_text, request
    ):
        basic_model = request.getfixturevalue(model_name)
        _contents = request.getfixturevalue(contents_name)

        basic_model.set_contents(_contents)
        result = basic_model.score_contents(input_text, return_spell_corrected=True)

        assert "spell_corrected" in result

    @pytest.mark.parametrize("input_text", sample_messages)
    @pytest.mark.parametrize(
        "model_name, contents_name",
        [
            ("stepwise_basic_model", "tagsets"),
            ("wmd_basic_model", "contents"),
        ],
    )
    def test_basic_model_score_with_empty_tags_returns_empty_scores(
        self, model_name, contents_name, input_text, request
    ):
        basic_model = request.getfixturevalue(model_name)
        _contents = request.getfixturevalue(contents_name)

        basic_model.set_contents([])
        assert len(basic_model.contents) == 0

        result = basic_model.score_contents(input_text, return_spell_corrected=False)
        scores = result["overall_scores"]
        assert len(scores) == 0

    @pytest.mark.parametrize(
        "input_text",
        sample_messages,
    )
    @pytest.mark.parametrize(
        "model_name, contents_name",
        [
            ("stepwise_basic_model", "tagsets"),
            ("wmd_basic_model", "contents"),
        ],
    )
    def test_basic_model_score_with_nonempty_tags_returns_nonempty_scores(
        self, model_name, contents_name, input_text, request
    ):
        basic_model = request.getfixturevalue(model_name)
        _contents = request.getfixturevalue(contents_name)

        basic_model.set_contents(_contents)
        assert len(basic_model.contents) == len(_contents)

        result = basic_model.score_contents(input_text, return_spell_corrected=True)
        scores = result["overall_scores"]
        spell_corrected = result["spell_corrected"]

        if len(spell_corrected) == 0:
            assert len(scores) == 0
        else:
            assert len(scores) == len(_contents)

    @pytest.mark.parametrize(
        "model_name, contents_name",
        [
            ("stepwise_weighted_model", "tagsets"),
            ("wmd_weighted_model", "contents"),
        ],
    )
    def test_resetting_contents_without_weights_is_allowed_with_warning(
        self, model_name, contents_name, content_weights, request
    ):
        weighted_model = request.getfixturevalue(model_name)
        _contents = request.getfixturevalue(contents_name)
        weighted_model.set_contents(_contents, weights=content_weights)
        with pytest.raises(UserWarning):
            weighted_model.set_contents([], weights=None)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        "basic_model_name, weighted_model_name, contents_name",
        [
            ("stepwise_basic_model", "stepwise_weighted_model", "tagsets"),
            ("wmd_basic_model", "wmd_weighted_model", "contents"),
        ],
    )
    @pytest.mark.parametrize("input_text", sample_messages)
    def test_basic_model_with_weights_returns_weighted_scores(
        self,
        basic_model_name,
        weighted_model_name,
        contents_name,
        content_weights,
        input_text,
        request,
    ):
        basic_model = request.getfixturevalue(basic_model_name)
        weighted_model = request.getfixturevalue(weighted_model_name)
        _contents = request.getfixturevalue(contents_name)

        weights = np.asarray(content_weights) / np.sum(content_weights)

        basic_model.set_contents(_contents, weights=None)
        unweighted_result = basic_model.score_contents(input_text)
        unweighted_scores = unweighted_result["overall_scores"]

        weighted_model.set_contents(_contents, weights=content_weights)
        weighted_result = weighted_model.score_contents(input_text)
        weighted_scores = weighted_result["overall_scores"]

        # Expects N=1.0
        for i, (u, w) in enumerate(zip(unweighted_scores, weighted_scores)):
            assert w == (u + weights[i]) / 2

    @pytest.mark.parametrize(
        "model_name, contents_name",
        [
            ("stepwise_weighted_model", "tagsets"),
            ("wmd_weighted_model", "contents"),
        ],
    )
    def test_weights_correctly_calculated_with_weights(
        self, model_name, contents_name, content_weights, request
    ):
        weighted_model = request.getfixturevalue(model_name)
        _contents = request.getfixturevalue(contents_name)

        weighted_model.set_contents(_contents, weights=content_weights)

        assert np.isclose(sum(weighted_model.content_weights), 1.0)
        assert np.allclose(
            np.asarray(weighted_model.content_weights),
            np.asarray(content_weights) / sum(content_weights),
        )


class TestStepwiseKeyedVectorScorer:
    @pytest.mark.parametrize(
        "input_text",
        sample_messages,
    )
    def test_return_tag_scores_flag(self, stepwise_basic_model, tagsets, input_text):
        stepwise_basic_model.set_contents(tagsets)
        result = stepwise_basic_model.score_contents(input_text, return_tag_scores=True)

        assert "tag_scores" in result
        assert len(result["overall_scores"]) == len(result["tag_scores"])

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_setting_contents_without_weights_sets_all_tag_attr(
        self,
        stepwise_basic_model,
        tagsets,
    ):
        stepwise_basic_model.set_contents(tagsets, weights=None)

        assert stepwise_basic_model.contents is not None
        assert stepwise_basic_model.content_vectors is not None
        assert stepwise_basic_model.content_weights is None

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        "input_text",
        sample_messages,
    )
    def test_basic_model_with_weights_returns_weighted_scores(
        self,
        stepwise_basic_model,
        stepwise_weighted_model,
        tagsets,
        content_weights,
        input_text,
    ):
        weights = np.asarray(content_weights) / np.sum(content_weights)

        stepwise_basic_model.set_contents(tagsets, weights=None)
        unweighted_result = stepwise_basic_model.score_contents(input_text)
        unweighted_scores = unweighted_result["overall_scores"]

        stepwise_weighted_model.set_contents(tagsets, weights=content_weights)
        weighted_result = stepwise_weighted_model.score_contents(input_text)
        weighted_scores = weighted_result["overall_scores"]

        # Expects N=1.0
        for i, (u, w) in enumerate(zip(unweighted_scores, weighted_scores)):
            assert w == (u + weights[i]) / 2

    def test_weights_correctly_calculated_with_weights(
        self, stepwise_weighted_model, tagsets, content_weights
    ):
        stepwise_weighted_model.set_contents(tagsets, weights=content_weights)

        assert np.isclose(sum(stepwise_weighted_model.content_weights), 1.0)
        assert np.allclose(
            np.asarray(stepwise_weighted_model.content_weights),
            np.asarray(content_weights) / sum(content_weights),
        )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_scores_with_weights_increase_rank(
        self,
        stepwise_basic_model,
        stepwise_weighted_model,
        tagsets,
        content_weights,
    ):
        input_text = "I love the outdoors. What should I pack for lunch?"

        stepwise_basic_model.set_contents(tagsets, weights=content_weights)
        unweighted_result = stepwise_basic_model.score_contents(input_text)
        unweighted_scores = unweighted_result["overall_scores"]
        ranks_mean_plus_weight = np.argsort(unweighted_scores)

        stepwise_weighted_model.set_contents(tagsets, weights=None)
        weighted_result = stepwise_weighted_model.score_contents(input_text)
        weighted_scores = weighted_result["overall_scores"]
        ranks_simple_mean = np.argsort(weighted_scores)

        assert np.argwhere(ranks_simple_mean == 1) <= np.argwhere(
            ranks_mean_plus_weight == 1
        )
        assert np.argwhere(ranks_simple_mean == 2) <= np.argwhere(
            ranks_mean_plus_weight == 2
        )


class TestWMDScorer:
    def test_scoring_without_setting_raises_error(self, wmd_basic_model):
        with pytest.raises(
            ValueError, match="Set contents with `self\.set_contents\(\)`"
        ):
            wmd_basic_model.score_contents("test message")

    def test_setting_contents_with_list_of_list_of_str(self, wmd_basic_model, tagsets):
        wmd_basic_model.set_contents(tagsets)
        assert wmd_basic_model.contents is not None

    def test_setting_contents_with_list_of_str(self, wmd_basic_model, contents):
        wmd_basic_model.set_contents(contents)
        assert wmd_basic_model.contents is not None

    def test_setting_contents_with_mixed_content_types_raises_error(
        self, wmd_basic_model, tagsets
    ):
        mixed_type_contents = tagsets.copy()
        mixed_type_contents[1] = " ".join(tagsets[1])
        mixed_type_contents[2] = " ".join(tagsets[2])

        with pytest.raises(TypeError):
            wmd_basic_model.set_contents(mixed_type_contents)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_setting_contents_without_weights_sets_all_tag_attr(
        self,
        wmd_basic_model,
        tagsets,
    ):
        wmd_basic_model.set_contents(tagsets, weights=None)

        assert wmd_basic_model.contents is not None
        assert isinstance(wmd_basic_model.wmd_index, WmdSimilarity)
        assert wmd_basic_model.content_weights is None

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_scores_with_weights_increase_rank(
        self,
        wmd_basic_model,
        wmd_weighted_model,
        tagsets,
        content_weights,
    ):
        input_text = "I love the outdoors. What should I pack for lunch?"

        wmd_basic_model.set_contents(tagsets, weights=content_weights)
        unweighted_result = wmd_basic_model.score_contents(input_text)
        unweighted_scores = unweighted_result["overall_scores"]
        ranks_mean_plus_weight = np.argsort(unweighted_scores)

        wmd_weighted_model.set_contents(tagsets, weights=None)
        weighted_result = wmd_weighted_model.score_contents(input_text)
        weighted_scores = weighted_result["overall_scores"]
        ranks_simple_mean = np.argsort(weighted_scores)

        assert np.argwhere(ranks_simple_mean == 1) <= np.argwhere(
            ranks_mean_plus_weight == 1
        )
        assert np.argwhere(ranks_simple_mean == 2) <= np.argwhere(
            ranks_mean_plus_weight == 2
        )
