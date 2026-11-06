import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.evaluate import asqa_inputs, evaluate_longform
from experiments.run import _json_atomic


def export_for_reader(payload, output):
    if payload.get("run_status") != "complete" or any(r.get("error") for r in payload["results"]):
        raise ValueError("Only complete, error-free generation runs can be evaluated")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    references, predictions = asqa_inputs(payload["results"])
    examples = []
    for key, ref in references.items():
        for i, pair in enumerate(ref["qa_pairs"]):


            examples.append({"id": f"{key}_{i}", "question": pair["question"], "context": predictions[key]})
    _json_atomic(output / "qa_inputs.json", {"data": examples})
    _json_atomic(output / "asqa_draws.json", {"dev": references})
    _json_atomic(output / "long_predictions.json", predictions)
    _json_atomic(output / "draw_manifest.json", [
        {"draw_id": r.get("draw_id", f"draw_{i:06d}"), "source_id": r["id"]}
        for i, r in enumerate(payload["results"])])
    return examples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--qa-predictions", help="JSON from asqa_reader.py or upstream RoBERTa (matching draw IDs)")
    p.add_argument("--scorer-seed", type=int, default=42)
    args = p.parse_args()
    payload = json.loads(Path(args.results).read_text())
    export_for_reader(payload, args.output_dir)
    if args.qa_predictions:
        qa = json.loads(Path(args.qa_predictions).read_text())
        metrics = evaluate_longform(payload["results"], qa, args.scorer_seed)
        metrics["qa_predictions_sha256"] = hashlib.sha256(Path(args.qa_predictions).read_bytes()).hexdigest()
        reader_metadata = Path(args.qa_predictions).with_suffix(".metadata.json")
        if reader_metadata.is_file():
            metrics["qa_reader_provenance"] = json.loads(reader_metadata.read_text())
        metrics["answer_stages"] = sorted({r.get("answer_stage", "unknown") for r in payload["results"]})
        _json_atomic(Path(args.output_dir) / "metrics.json", metrics)

        _json_atomic(Path(args.output_dir) / "scored_results.json", {**payload, "metrics": metrics})
        print(json.dumps(metrics, indent=2))
    else:
        print("Exported QA inputs. No Disambig-F1 or DR calculated until actual QA-reader predictions are supplied.")


if __name__ == "__main__":
    main()
