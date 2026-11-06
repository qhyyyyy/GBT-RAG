import json
from pathlib import Path
import numpy as np
import pytest
from data.loader import load_asqa, _sample_list, load_dataset
from experiments.evaluate import evaluate_longform, asqa_inputs, hotpot_f1
from experiments.run import run_method, _load_config
from scripts.score_asqa import export_for_reader


def example(question="Q?"):
    return {"ambiguous_question": question,
            "annotations": [{"long_answer": "Paris is in France."}, {"long_answer": "The city is Paris, France."}],
            "qa_pairs": [{"question": "Where?", "short_answers": ["Paris", "City of Paris"],
                          "context": "GOLD_CONTEXT_SENTINEL", "wikipage": "Paris"}]}


def test_asqa_official_mapping_and_no_gold_context(tmp_path):
    file = tmp_path / "asqa.json"
    file.write_text(json.dumps({"dev": {"original-id": example()}}))
    rows = load_asqa(str(file), n_samples=5, replacement=True)
    assert len(rows) == 5
    assert all(row["id"] == "original-id" for row in rows)
    assert rows[0]["context"] == []
    assert len(rows[0]["qa_pairs"]) == 1
    assert len(rows[0]["qa_pairs"][0]["short_answers"]) == 2


def test_question_sampling_repeats_and_manifest(cfg, tmp_path):
    file = tmp_path / "asqa.json"
    file.write_text(json.dumps({"dev": {str(i): example(str(i)) for i in range(3)}}))
    cfg.data.name = "asqa"; cfg.data.path = str(file)
    cfg.data.n_samples = 500; cfg.data.sample_with_replacement = True
    cfg.data.expected_pool_size = 3; cfg.retrieval.scope = "global"
    a, b = load_dataset(cfg), load_dataset(cfg)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert len(a) == len({r["draw_id"] for r in a}) == 500
    assert len({r["id"] for r in a}) == 3
    cfg.data.expected_pool_size = 948
    with pytest.raises(ValueError, match="Expected 948"):
        load_dataset(cfg)


def test_labels_never_enter_generator(cfg, tmp_path):
    cfg.data.name = "asqa"
    sample = {"id": "s", "draw_id": "draw_000000", "draw_index": 0,
              "question": "Q?", "answer": "SECRET_GOLD", "answer_aliases": ["SECRET_GOLD", "GOLD2"],
              "qa_pairs": [{"question": "Where?", "short_answers": ["SECRET_LABEL"]}],
              "short_answers": ["SECRET_LABEL"], "context": [], "task_type": "longform", "dataset": "asqa"}
    class Generator:
        def run(self, question, docs, task_type):
            assert question == "Q?" and docs == [] and task_type == "longform"
            return {"answer": "generated"}
    run_method("gbtrag", Generator(), [sample], cfg, tmp_path)
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["results"][0]["gold"] == "SECRET_GOLD"
    assert saved["metrics"]["evaluation_status"] == "pending_official_qa_reader"
    assert "dr_score" not in saved["metrics"]
    inputs = export_for_reader(saved, tmp_path / "qa")
    assert all(set(row) == {"id", "question", "context"} for row in inputs)
    assert "SECRET_" not in json.dumps(inputs)


def test_hotpot_yes_no_exception():
    assert hotpot_f1("yes indeed", "yes") == 0
    assert hotpot_f1("yes", "yes") == 1


def test_official_asqa_adapter_preserves_draw_multiplicity(monkeypatch):
    import nltk

    monkeypatch.setattr(nltk, "sent_tokenize", lambda text: [text])
    rows = []
    for i in range(2):
        rows.append({"id": "same-source", "draw_id": f"draw_{i:06d}", "answer": "Paris is in France.",
                     "answer_aliases": ["Paris is in France.", "The city is Paris, France."],
                     "qa_pairs": [{"question": "Where?", "short_answers": ["Paris", "City of Paris"]}],
                     "task_type": "longform"})
    refs, preds = asqa_inputs(rows)
    assert len(preds) == 2
    qa = {"draw_000000_0": "Paris", "draw_000001_0": "wrong"}
    metrics = evaluate_longform(rows, qa)
    assert metrics["disambig_f1"] == 0.5
    assert metrics["dr_score"] == pytest.approx((metrics["disambig_f1"] * metrics["rouge_l"]) ** 0.5)
    from experiments.evaluate import official_asqa_module
    np.random.seed(42)
    expected = official_asqa_module().compute_all_scores({"answers": preds, "qa": qa}, refs)
    assert metrics["official_asqa_scores_percent"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="IDs mismatch"):
        evaluate_longform(rows, {"draw_000000_0": "Paris"})


def test_actual_overrides_win_over_dataset_defaults():
    cfg = _load_config("configs/default.yaml", ["data.n_samples=7", "gbtrag.coverage_threshold=0.6"], dataset="asqa")
    assert cfg.data.n_samples == 7
    assert cfg.data.sample_with_replacement
    assert cfg.gbtrag.coverage_threshold == 0.6
    assert cfg.gbtrag.max_rounds == 3


def test_failed_sample_is_saved_without_success_metrics(cfg, tmp_path):
    sample = {"id": "s", "draw_id": "draw_0", "draw_index": 0, "question": "Q", "answer": "a", "answer_aliases": ["a"], "context": []}
    class Broken:
        def run(self, *args, **kwargs): raise RuntimeError("service unavailable")
    with pytest.raises(RuntimeError, match="partial output saved"):
        run_method("gbtrag", Broken(), [sample], cfg, tmp_path)
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["run_status"] == "failed"
    assert saved["metrics"]["evaluation_status"] == "incomplete_generation"
    assert "f1" not in saved["metrics"]


def test_completed_resume_preserves_existing_metrics(cfg, tmp_path):
    sample = {"id": "s", "draw_id": "draw_0", "draw_index": 0, "question": "Q", "answer": "Paris", "answer_aliases": ["Paris"], "context": []}
    class Generator:
        def run(self, *args, **kwargs): return {"answer": "Paris"}
    run_method("gbtrag", Generator(), [sample], cfg, tmp_path)
    path = next(tmp_path.glob("*.json"))
    before = path.read_bytes()
    metrics = run_method("gbtrag", Generator(), [sample], cfg, tmp_path, resume=str(path))
    assert metrics["f1"] == 1
    assert path.read_bytes() == before


def test_average_of_sample_DR_is_not_recomputed_from_average_components(tmp_path):
    from experiments.summarize import summarize
    paths = []
    for seed, df1, rouge in [(42, 0.25, 1.0), (99, 1.0, 0.25)]:
        cfg = {"data": {"seed": seed}}
        run = {"model": "same", "method": "gbtrag", "dataset": "asqa", "code_sha256": "same",
               "run_status": "complete", "sampling_seed": seed, "config": cfg,
               "metrics": {"evaluation_status": "complete", "metric_protocol": "asqa_official",
                           "dr_score": 0.5, "disambig_f1": df1, "rouge_l": rouge}}
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps(run))
        paths.append(path)
    result = summarize(paths)
    assert result["metrics"]["dr_score"] == 0.5
    assert result["metrics"]["dr_score"] != pytest.approx(
        (result["metrics"]["disambig_f1"] * result["metrics"]["rouge_l"]) ** 0.5)
