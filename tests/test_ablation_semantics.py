from dataclasses import replace
import sys

import pytest

from tests.test_pipeline import pipeline


def test_single_round_means_initial_tree_without_updates(cfg, monkeypatch):
    cfg.gbtrag.single_round = True
    cfg.gbtrag.max_rounds = 3
    model = pipeline(cfg, monkeypatch, [])
    result = model.run("Q?", [{"title": "A", "text": "alpha evidence."}], "longform")
    assert result["num_boosting_rounds"] == 0
    assert len(result["trees"]) == len(result["round_trace"]) == 1
    assert result["round_trace"][0]["m"] == 0
    assert not result["round_logs"]


@pytest.mark.parametrize("all_leaves, expected_size", [(False, 1), (True, 2)])
def test_residual_tree_ablation_retrieves_covered_leaves_too(cfg, monkeypatch, all_leaves, expected_size):
    cfg.gbtrag.no_residual = all_leaves
    cfg.gbtrag.max_rounds = 1
    model = pipeline(cfg, monkeypatch, ["better"])

    def coverage(answer, dims):
        return [replace(d, sub_aspect_sims=[0.8 if s == "alpha" else
                        (0.3 if answer == "better" else 0.2) for s in d.sub_aspects])
                for d in dims]

    monkeypatch.setattr(model.residual_engine, "compute_coverage", coverage)
    result = model.run("Q?", [{"title": "A", "text": "alpha evidence."}], "longform")
    assert result["num_boosting_rounds"] == 1
    assert result["trees"][1]["num_leaves"] == expected_size
    log = result["round_logs"][0]
    assert log["active_leaf_ids"] == ["s1"]
    assert log["coverage_loss_after"] < log["coverage_loss_before"]
    if all_leaves:
        assert log["residual_restricted"] is False
        assert set(log["tree_leaf_ids"]) == {"s0", "s1"}


def test_unspecified_paper_critique_ablation_is_not_mislabeled(cfg, monkeypatch, capsys):
    from experiments.run import main
    monkeypatch.setattr(sys, "argv", ["run.py", "--ablation", "no_coverage_loss", "--dry-run"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "text-only critique" in capsys.readouterr().err


def test_word_overlap_has_explicit_diagnostic_name(cfg, monkeypatch, capsys):
    import experiments.run as runner
    monkeypatch.setattr(sys, "argv", ["run.py", "--ablation", "word_overlap_coverage", "--dry-run"])
    monkeypatch.setattr(runner, "load_dataset", lambda _: [{"id": "one"}])
    runner.main()
    assert "coverage_fn: word_overlap" in capsys.readouterr().out
