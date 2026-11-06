import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ColumnRows(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self]
        return super().__getitem__(key)


def run_reader(input_path, model_path, output_path, device="cpu", batch_size=8):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    from third_party.transformers_qa.utils_qa import postprocess_qa_predictions
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError("Provide the local ASQA roberta-squad checkpoint directory")
    examples = ColumnRows(json.loads(Path(input_path).read_text())["data"])
    if not examples:
        raise ValueError("No QA examples")
    if len({x["id"] for x in examples}) != len(examples):
        raise ValueError("QA IDs must be unique; preserve bootstrap draw IDs")
    if any(set(x) != {"id", "question", "context"} for x in examples):
        raise ValueError("QA inference input must contain only id/question/generated context")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, local_files_only=True)
    if not tokenizer.is_fast or tokenizer.padding_side != "right":
        raise ValueError("Official RoBERTa evaluation requires fast tokenization and right padding")
    model = AutoModelForQuestionAnswering.from_pretrained(str(model_path), local_files_only=True)
    if model.config.model_type != "roberta":
        raise ValueError("Use the ASQA RoBERTa-SQuAD2 checkpoint, not a different QA-reader family")
    model.to(device).eval()
    encoded = tokenizer([x["question"].lstrip() for x in examples],
                        [x["context"] for x in examples], truncation="only_second",
                        max_length=384, stride=128, return_overflowing_tokens=True,
                        return_offsets_mapping=True, padding="max_length")
    sample_map = encoded.pop("overflow_to_sample_mapping")
    offsets = encoded.pop("offset_mapping")
    features = []
    for i, sample_index in enumerate(sample_map):
        sequence_ids = encoded.sequence_ids(i)
        features.append({"example_id": examples[sample_index]["id"],
                         "offset_mapping": [span if sequence_ids[k] == 1 else None for k, span in enumerate(offsets[i])]})
    keys = [key for key in tokenizer.model_input_names if key in encoded]
    starts, ends = [], []
    with torch.inference_mode():
        for begin in range(0, len(features), batch_size):
            tensors = {key: torch.tensor(encoded[key][begin:begin + batch_size], device=device) for key in keys}
            output = model(**tensors)
            starts.append(output.start_logits.cpu().numpy())
            ends.append(output.end_logits.cpu().numpy())
    predictions = postprocess_qa_predictions(
        examples, features, (np.concatenate(starts), np.concatenate(ends)),
        version_2_with_negative=True, n_best_size=20, max_answer_length=30,
        null_score_diff_threshold=0.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(predictions), ensure_ascii=False, indent=2) + "\n")
    config_path = model_path / "config.json"
    provenance = {"qa_model": str(model_path), "qa_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  "max_seq_length": 384, "doc_stride": 128, "n_best_size": 20,
                  "max_answer_length": 30, "version_2_with_negative": True,
                  "null_score_diff_threshold": 0.0, "num_questions": len(examples),
                  "num_features": len(features), "uses_gold_answers": False}
    output_path.with_suffix(".metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return dict(predictions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--qa-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    result = run_reader(args.input, args.qa_model, args.output, args.device, args.batch_size)
    print(f"Saved {len(result)} actual QA predictions to {args.output}")


if __name__ == "__main__":
    main()
