from dataclasses import replace
import inspect
import json
import numpy as np
import pytest

from src.gbtrag import GBTRAG
from src.llm import Embedder
from src.residual import Residual
from src.tree import RetrievalDecisionTree


class FakeLLM:

    context = ""
    def __init__(self): self.reset_call_count()
    def reset_call_count(self): self.calls = 0; self.prompts = []
    def __call__(self, prompt, system=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        return "alpha evidence."
    def json(self, prompt, system=None):
        self.calls += 1
        if '"query"' in prompt or '"search_query"' in prompt:
            return {"query": "alpha evidence"}
        if '"relevant"' in prompt:
            return {"relevant": True, "evidence": "alpha evidence."}
        return {"dimensions": [{"name": "facet", "ideal": "facet", "sub_aspects": ["alpha", "beta"]}]}
    def count_calls(self): return self.calls
    def get_call_stats(self): return {"total_calls": self.calls, "api_calls": 0, "cache_hits": 0}


def pipeline(cfg, monkeypatch, decisions):
    llm = FakeLLM()
    model = GBTRAG(cfg, llm=llm, embedder=Embedder("test", force_mock=True))
    monkeypatch.setattr(model, "_phase0_draft_critique", lambda *a, **k: ([{"name": "facet", "sub_aspects": ["alpha", "beta"]}], "", False, 0))
    monkeypatch.setattr(model, "_ground_dimensions", lambda dims, *a, **k: dims)
    def coverage(answer, dims):
        value = {"alpha evidence.": 0.2, "better": 0.3, "best": 0.4, "bad": 0.1}.get(answer, 0.2)
        return [replace(d, coverage=value, sub_aspect_sims=[value] * len(d.sub_aspects)) for d in dims]
    monkeypatch.setattr(model.residual_engine, "compute_coverage", coverage)
    answers = iter(decisions)
    monkeypatch.setattr(model, "_compose", lambda *args: next(answers))
    return model


def test_executes_all_M_updates(cfg, monkeypatch):
    cfg.gbtrag.max_rounds = 3
    model = pipeline(cfg, monkeypatch, ["better", "best", "bad"])
    result = model.run("A question?", [{"text": "alpha evidence.", "title": "A"}], "longform")
    assert result["num_boosting_rounds"] == 3
    assert len(result["round_trace"]) == 4
    assert result["detailed_answer"] == "best"
    assert [x["decision"] for x in result["round_logs"]] == ["accept", "accept", "rollback"]
    assert result["final_loss"] == pytest.approx(0.5)
    assert all(x["coverage_loss_after"] <= x["coverage_loss_before"] for x in result["round_logs"])


def test_two_rollbacks_do_not_terminate(cfg, monkeypatch):
    model = pipeline(cfg, monkeypatch, ["bad", "bad", "better"])
    result = model.run("Q?", [{"text": "alpha evidence.", "title": "A"}], "longform")
    assert result["num_boosting_rounds"] == 3
    assert result["detailed_answer"] == "better"


def test_candidate_uses_expanded_coordinate_baseline(cfg, monkeypatch):
    cfg.gbtrag.max_rounds = 1
    cfg.gbtrag.enable_dimension_discovery = True
    model = pipeline(cfg, monkeypatch, ["better"])
    monkeypatch.setattr(model.residual_engine, "discover", lambda *a, **k: [{"name": "new", "sub_aspects": ["gamma"]}])

    monkeypatch.setattr(model, "_expand", lambda dims, proposals, r: (dims + model.residual_engine.init_dimensions(proposals), ["s2"]))
    result = model.run("Q?", [{"text": "alpha evidence.", "title": "A"}], "longform")
    log = result["round_logs"][0]
    assert log["coverage_loss_before"] == pytest.approx(3 * 0.45)
    assert log["candidate_loss"] == pytest.approx(3 * 0.35)
    assert log["decision"] == "accept"
    assert set(log["active_leaf_ids"]) == {"s0", "s1", "s2"}


def test_generation_signatures_cannot_receive_labels():
    assert "short_answers" not in inspect.signature(GBTRAG.run).parameters
    assert "short_answers" not in inspect.signature(GBTRAG._phase7_compress).parameters


def test_compression_keeps_raw_diagnostics(cfg, monkeypatch):
    cfg.gbtrag.compress_output = True
    cfg.gbtrag.max_rounds = 0
    model = pipeline(cfg, monkeypatch, [])
    monkeypatch.setattr(model, "_phase7_compress", lambda *a: ("compressed", True))
    result = model.run("Q?", [{"text": "alpha evidence.", "title": "A"}], "longform")
    assert result["answer"] == "compressed"
    assert result["a_final_raw"] == result["detailed_answer"] == "alpha evidence."
    assert result["round_trace"][0]["answer_raw"] == "alpha evidence."


def test_dynamic_tree_does_not_truncate_to_12_or_max_children(cfg):
    llm, emb = FakeLLM(), Embedder("test", force_mock=True)
    from src.residual import GBTRAGResidual
    dims = GBTRAGResidual(emb).init_dimensions([
        {"name": f"facet{i}", "sub_aspects": [f"fact {i}-{j}" for j in range(4)]} for i in range(5)])
    tree = RetrievalDecisionTree(0, llm, emb, None, max_children=4)
    tree.build("Q", 2, dimensions=dims, sub_aspects_per_dim=3)
    assert len(tree.get_leaves()) == 20
    assert tree.stats()["max_depth"] == 2


def test_rerank_happens_before_extraction():
    llm, emb = FakeLLM(), Embedder("test", force_mock=True)
    class Retriever:
        def search(self, *a, **k): return [{"text": "wrong", "title": "X"}, {"text": "alpha evidence.", "title": "Y"}]
    class Reranker:
        def rerank(self, query, docs, top_k): return docs[1:]
    tree = RetrievalDecisionTree(1, llm, emb, Retriever())
    from src.residual import GBTRAGResidual
    dims = GBTRAGResidual(emb).init_dimensions([{"name": "a", "sub_aspects": ["a"]}])
    tree.build("Q", 2, dimensions=dims)
    tree.execute(reranker=Reranker())
    assert tree.get_leaves()[0].evidence == "alpha evidence."
