import numpy as np
import pytest
from src.llm import Embedder
from src.residual import GBTRAGResidual, Dimension, _chunk_answer


class Vectors:
    embed_dim = 2
    def encode(self, texts):
        if isinstance(texts, str): texts = [texts]
        mapping = {"covered": [1, 0], "missed": [0, 1], "answer": [1, 0]}
        return np.asarray([mapping[t] for t in texts], dtype=np.float32)


def test_leaf_support_is_not_dimension_average():
    engine = GBTRAGResidual(Vectors(), tau=0.4)
    dims = engine.init_dimensions([{"name": "facet", "ideal": "DO NOT EMBED THIS",
                                    "sub_aspects": ["covered", "missed"]}])
    scored = engine.compute_coverage("answer", dims)
    assert scored[0].coverage == 0.5
    assert dims[0].sub_aspect_sims == []
    assert [g.leaf_id for g in engine.backpropagate(scored)] == ["s1"]
    assert engine.compute_loss(scored) == pytest.approx(0.4)
    scored[0].leaf_weights = [1, 0.5]
    assert engine.compute_loss(scored) == pytest.approx(0.2)
    assert engine.active_dimensions(scored)[0].leaf_ids == ["s1"]


def test_empty_answer_clears_every_coordinate():
    engine = GBTRAGResidual(Vectors())
    dims = engine.init_dimensions([{"name": "x", "sub_aspects": ["covered", "missed"]}])
    covered = engine.compute_coverage("answer", dims)
    empty = engine.compute_coverage("", covered)
    assert empty[0].sub_aspect_sims == [0, 0]
    assert engine.compute_loss(empty) == pytest.approx(1.3)
    assert _chunk_answer("Paris") == ["Paris"]


def test_dead_leaf_pruning_uses_prior_evidence_per_leaf():
    engine = GBTRAGResidual(Vectors())
    dims = engine.compute_coverage("", engine.init_dimensions([
        {"name": "x", "sub_aspects": ["covered", "missed"]}]))
    kept, removed = engine.prune_dead_leaves(dims, ["covered"])
    assert removed == ["s1"]
    assert kept[0].leaf_ids == ["s0"]


def test_missing_model_is_not_a_random_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        Embedder(str(tmp_path / "missing"))
    explicit = Embedder("synthetic", force_mock=True)
    assert np.array_equal(explicit.encode("same"), explicit.encode("same"))
    assert explicit.is_mock
