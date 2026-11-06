from dataclasses import replace
import json
import numpy as np
import pytest
from tests.test_pipeline import pipeline


def test_even_a_small_loss_increase_is_rejected(cfg, monkeypatch):
    cfg.gbtrag.max_rounds = 1
    model = pipeline(cfg, monkeypatch, ["tiny_regression"])
    base_coverage = model.residual_engine.compute_coverage
    def score(answer, dims):
        if answer == "tiny_regression":
            return [replace(d, coverage=0.19999, sub_aspect_sims=[0.19999] * len(d.sub_aspects)) for d in dims]
        return base_coverage(answer, dims)
    monkeypatch.setattr(model.residual_engine, "compute_coverage", score)
    out = model.run("Q", [{"title": "A", "text": "alpha evidence."}], "longform")
    assert out["round_logs"][0]["decision"] == "rollback"


def test_stagnation_only_after_acceptance(cfg, monkeypatch):
    model = pipeline(cfg, monkeypatch, ["alpha evidence.", "better", "best"])
    out = model.run("Q", [{"title": "A", "text": "alpha evidence."}], "longform")
    assert out["num_boosting_rounds"] == 1
    assert out["stop_reason"] == "accepted_stagnation"


def test_qa_span_and_null_use_upstream_helper():
    from scripts.asqa_reader import ColumnRows
    from third_party.transformers_qa.utils_qa import postprocess_qa_predictions
    examples = ColumnRows([{"id": "x", "context": "Paris city", "question": "Where?"}])
    features = [{"example_id": "x", "offset_mapping": [None, (0, 5), (6, 10), None]}]
    result = postprocess_qa_predictions(examples, features,
        (np.array([[0., 8., 1., 0.]]), np.array([[0., 8., 1., 0.]])), version_2_with_negative=True)
    assert result["x"] == "Paris"
    result = postprocess_qa_predictions(examples, features,
        (np.array([[20., 8., 1., 0.]]), np.array([[20., 8., 1., 0.]])), version_2_with_negative=True)
    assert result["x"] == ""


def test_offline_round_compression_does_not_touch_labels(cfg, tmp_path):
    from experiments.postprocess import postprocess_rounds
    from omegaconf import OmegaConf
    cfg.data.name = "asqa"
    row = {"id": "source", "draw_id": "draw_0", "question": "Q?", "gold": "SECRET_GOLD",
           "answer_aliases": ["SECRET_GOLD", "SECRET_ALIAS"],
           "qa_pairs": [{"question": "subQ", "short_answers": ["SECRET_LABEL"]}],
           "round_trace": [{"m": 0, "answer_raw": "raw answer", "coverage_loss": 0.3,
                            "active_set_size": 1, "cumulative_leaf_retrievals": 2}]}
    class Recorder:
        context = ""
        def __call__(self, prompt, **kwargs):
            assert "SECRET" not in prompt + self.context
            return "compressed answer"
    out = postprocess_rounds({"config": OmegaConf.to_container(cfg), "run_status": "complete", "results": [row]},
                             Recorder(), tmp_path, [0, 2])
    assert len(out) == 2
    saved = json.loads(out[1].read_text())
    assert saved["results"][0]["actual_trace_position"] == 0
    assert row["round_trace"][0]["answer_raw"] == "raw answer"
