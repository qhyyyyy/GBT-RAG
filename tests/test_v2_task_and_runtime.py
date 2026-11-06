from pathlib import Path
import json
import pytest
import numpy as np

from tests.test_pipeline import FakeLLM
from src.gbtrag import GBTRAG
from src.llm import Embedder
from src.residual import GBTRAGResidual
from src.tree import RetrievalDecisionTree


def test_shortform_auto_uses_grounded_generic_prompt(cfg, monkeypatch):
    cfg.gbtrag.max_rounds = 0
    class Recorder(FakeLLM):
        def json(self, prompt, system=None):
            self.prompts.append(prompt)
            return super().json(prompt, system)
    llm = Recorder()
    model = GBTRAG(cfg, llm=llm, embedder=Embedder("synthetic", force_mock=True))
    monkeypatch.setattr(model, "_phase0_draft_critique", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ASQA critique used for shortform")))
    monkeypatch.setattr(model, "_extract_short_answer", lambda *a: "short span")
    out = model.run("Who is the author?", [{"title": "Book", "text": "alpha evidence."}], "shortform")
    assert out["phase0_mode_used"] == "grounded_decompose"
    generic = [p for p in llm.prompts if "Produce exactly 4 dimensions" in p]
    assert len(generic) == 1
    assert "[Book] alpha evidence." in generic[0]
    assert "same short-answer" in " ".join(llm.prompts).lower() or "This is a short-answer question" in generic[0]
    assert not out["phase0_critique_used_retrieval_evidence"]


def test_short_answer_discovery_scope_and_array_schema():
    emb = Embedder("synthetic", force_mock=True)
    engine = GBTRAGResidual(emb)
    dims = engine.init_dimensions([{"name": "a", "sub_aspects": ["a"]}])
    class Recorder:
        def json(self, prompt):
            assert "SAME short-answer question" in prompt
            return [{"name": "hop", "sub_aspects": ["needed bridge fact"]}]
    assert engine.discover("answer", dims, Recorder(), task_type="shortform")[0]["name"] == "hop"


def test_prior_leaf_queries_trigger_alternative_query():
    emb = Embedder("synthetic", force_mock=True)
    dims = GBTRAGResidual(emb).init_dimensions([{"name": "a", "sub_aspects": ["bridge fact"]}])
    class Recorder(FakeLLM):
        def json(self, prompt, system=None):
            assert "Previous query: earlier keywords" in prompt
            assert "previous retrieved text" in prompt
            return {"query": "new alternative keywords"}
    tree = RetrievalDecisionTree(1, Recorder(), emb, None,
            query_history={"s0": {"query": "earlier keywords", "passages": "previous retrieved text"}})
    tree.build("Original question", 2, dimensions=dims)
    assert tree.get_leaves()[0].search_query == "new alternative keywords"


def test_embedding_cache_separates_roles_and_returns_copies(monkeypatch):
    emb = Embedder("synthetic", force_mock=True, cache_size=2)
    original = emb._encode_uncached
    calls = []
    def record(texts, batch_size, role):
        calls.append((list(texts), role))
        return original(texts, batch_size, role)
    monkeypatch.setattr(emb, "_encode_uncached", record)
    first = emb.encode(["one", "one", "two"])
    expected = first[0].copy()
    first[0] *= 0
    assert np.array_equal(emb.encode("one")[0], expected)
    emb.encode("one", role="passage")
    assert calls == [(["one", "two"], "symmetric"), (["one"], "passage")]


def test_retrieval_failure_is_not_silently_scored_as_success():
    emb = Embedder("synthetic", force_mock=True)
    dims = GBTRAGResidual(emb).init_dimensions([{"name": "a", "sub_aspects": ["a"]}])
    class Broken:
        def search(self, *a, **k): raise RuntimeError("index IO failed")
    tree = RetrievalDecisionTree(0, FakeLLM(), emb, Broken())
    tree.build("Q", 2, dimensions=dims)
    with pytest.raises(RuntimeError, match="index IO failed"):
        tree.execute()


def test_interrupt_saves_completed_prefix(cfg, tmp_path):
    from experiments.run import run_method
    sample = {"id": "s", "draw_id": "draw_0", "draw_index": 0, "question": "Q", "answer": "Paris", "answer_aliases": ["Paris"], "context": []}
    class Interrupted:
        def run(self, *a, **k): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_method("gbtrag", Interrupted(), [sample], cfg, tmp_path)
    output = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert output["run_status"] == "interrupted"
    assert output["results"] == []
    assert output["metrics"]["evaluation_status"] == "incomplete_generation"
