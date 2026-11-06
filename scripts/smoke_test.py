import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run import _load_config, run_method
from scripts.score_asqa import export_for_reader
from src.gbtrag import GBTRAG
from src.llm import Embedder


class ScriptedTestLLM:
    context = ""
    def __init__(self): self.reset_call_count()
    def reset_call_count(self): self.calls = 0
    def __call__(self, prompt, system=None, **kwargs):
        self.calls += 1
        return "Alpha evidence."
    def json(self, prompt, system=None):
        self.calls += 1
        if '"new_dimensions"' in prompt:
            return {"new_dimensions": []}
        if '"sub_aspects"' in prompt:
            return {"dimensions": [{"name": "synthetic facet", "sub_aspects": ["Alpha", "Beta"]}]}
        if '"relevant"' in prompt:
            return {"relevant": True, "evidence": "Alpha evidence."}
        if '"query"' in prompt:
            return {"query": "Alpha evidence"}
        raise ValueError("Unknown synthetic-test prompt")
    def get_call_stats(self):
        return {"total_calls": self.calls, "api_calls": 0, "cache_hits": 0, "synthetic": True}
    def count_calls(self): return self.calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="tmp/synthetic_smoke")
    args = p.parse_args()
    cfg = _load_config(dataset="asqa")
    cfg.retrieval.scope = "candidate_debug"
    cfg.reranker.enabled = False
    cfg.data.expected_pool_size = None
    cfg.gbtrag.enable_dead_leaf_pruning = False
    cfg.experiment.cache_dir = None
    cfg.embedding.model_path = "EXPLICIT_SYNTHETIC_TEST_DOUBLE"
    sample = {"id": "synthetic", "draw_id": "draw_000000", "draw_index": 0,
              "question": "Which synthetic fact is stated?", "answer": "Reference Alpha.",
              "answer_aliases": ["Reference Alpha.", "Reference Beta."],
              "short_answers": ["Alpha"],
              "qa_pairs": [{"question": "What is the fact?", "short_answers": ["Alpha"]}],
              "context": [{"title": "Synthetic document", "text": "Alpha evidence."}],
              "task_type": "longform", "dataset": "asqa"}
    method = GBTRAG(cfg, llm=ScriptedTestLLM(), embedder=Embedder("synthetic", force_mock=True))
    run_method("gbtrag", method, [sample], cfg, args.output_dir, output_suffix="SYNTHETIC_ONLY")
    files = list(Path(args.output_dir).glob("*SYNTHETIC_ONLY.json"))
    latest = max(files, key=lambda x: x.stat().st_mtime_ns)
    payload = json.loads(latest.read_text())
    payload["validation_mode"] = "synthetic_test_doubles_not_a_benchmark_run"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    export_for_reader(payload, Path(args.output_dir) / "qa_export")
    assert payload["run_status"] == "complete"
    assert payload["results"][0]["answer_stage"] == "post_compression"
    assert "dr_score" not in payload["metrics"]
    print("PASS: loading, tree execution, final compression, saving, and QA input export")
    print("Synthetic components only. No real-model accuracy was measured.")


if __name__ == "__main__":
    main()
