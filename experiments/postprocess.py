import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from experiments.run import _json_atomic
from src.llm import LLM
from src.prompts import COMPRESSION_PROMPT, fmt


def postprocess_rounds(payload, llm, output_dir, rounds):
    if payload.get("run_status") != "complete":
        raise ValueError("Round postprocessing requires a complete generation run")
    cfg = OmegaConf.create(payload["config"])
    output_dir = Path(output_dir)
    paths = []
    original_context = getattr(llm, "context", "")
    try:
        for m in rounds:
            rows = []
            for source in payload["results"]:
                trace = source.get("round_trace", [])
                available = [row for row in trace if row["m"] <= m]
                if not available:
                    raise ValueError("No saved A0/round answers; old runs without checkpoints cannot be reconstructed")
                state = max(available, key=lambda row: row["m"])
                entities = "\n".join(f'- "{e}"' for e in source.get("verified_entities", []))
                llm.context = (f"Original question: {source['question']}\n"
                               f"Verified named entities (preserve their surface forms):\n{entities or '(none)'}")
                answer = llm(fmt(COMPRESSION_PROMPT, A_final=state["answer_raw"],
                                 target_words=cfg.gbtrag.compress_target_words,
                                 structure_hint=cfg.gbtrag.compress_structure_hint),
                             max_tokens=cfg.llm.final_max_tokens).strip()
                for prefix in ["compressed answer:", "answer:"]:
                    if answer.lower().startswith(prefix):
                        answer = answer[len(prefix):].strip()
                if not answer:
                    raise RuntimeError("Round compression returned empty output")
                rows.append({**source, "answer": answer, "answer_stage": "post_compression",
                             "a_final_raw": state["answer_raw"], "a_compressed": answer,
                             "requested_trace_position": m, "actual_trace_position": state["m"],
                             "trace_diagnostics": {key: state[key] for key in
                                 ["coverage_loss", "active_set_size", "cumulative_leaf_retrievals"]}})
            output = {**payload, "results": rows, "trace_position": m,
                      "postprocess_only": True, "metrics": {"evaluation_status": "pending_official_qa_reader"}}
            path = output_dir / f"round_m{m}.json"
            _json_atomic(path, output)
            paths.append(path)
    finally:
        llm.context = original_context
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rounds", default="0,1,2,3")
    p.add_argument("--base-url", help="Override endpoint only; keep original model/decoding for comparability")
    args = p.parse_args()
    payload = json.loads(Path(args.results).read_text())
    cfg = OmegaConf.create(payload["config"])
    if cfg.data.name != "asqa":
        raise ValueError("This postprocessor implements the ASQA compression protocol")
    llm = LLM(args.base_url or cfg.llm.base_url, cfg.llm.model, cfg.llm.temperature,
              cfg.llm.max_tokens, cfg.llm.timeout, cfg.experiment.cache_dir, cfg.llm.seed, cfg.llm.max_retries)
    paths = postprocess_rounds(payload, llm, args.output_dir, [int(x) for x in args.rounds.split(",")])
    print("\n".join(str(path) for path in paths))
    print("No retrieval was repeated. QA-reader scoring is pending for these post-compression outputs.")


if __name__ == "__main__":
    main()
