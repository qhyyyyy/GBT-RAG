import argparse
import importlib.util
import json
import math
import re
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalize_answer(text):
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    p, g = normalize_answer(prediction).split(), normalize_answer(ground_truth).split()
    if not p or not g:
        return float(p == g)
    shared = sum((Counter(p) & Counter(g)).values())
    return 2 * shared / (len(p) + len(g))


def hotpot_f1(prediction, ground_truth):
    p, g = normalize_answer(prediction), normalize_answer(ground_truth)
    if (p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}) and p != g:
        return 0.0

    if not p or not g:
        return 0.0
    return f1_score(prediction, ground_truth)


def compute_score_with_aliases(prediction, answer_aliases, metric_fn):
    return max((metric_fn(prediction, alias) for alias in answer_aliases), default=0.0)


def evaluate_all(results):
    f1s, ems = [], []
    for result in results:
        pred = result.get("answer", "")
        aliases = result.get("answer_aliases", [result.get("gold", "")])
        aliases = list(dict.fromkeys([result.get("gold", "")] + list(aliases)))
        metric = hotpot_f1 if result.get("dataset") == "hotpotqa" else f1_score
        f1s.append(compute_score_with_aliases(pred, aliases, metric))
        ems.append(compute_score_with_aliases(pred, aliases, exact_match_score))
    return {"n": len(results), "f1": float(np.mean(f1s)) if f1s else 0.0,
            "em": float(np.mean(ems)) if ems else 0.0,
            "metric_protocol": "official_hotpot_answer_or_musique_alias_f1",
            "evaluation_status": "complete"}


def official_asqa_module():
    path = ROOT / "third_party/asqa/scoring.py"
    spec = importlib.util.spec_from_file_location("gbtrag_official_asqa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def asqa_inputs(results):

    references, predictions = {}, {}
    for i, row in enumerate(results):
        key = str(row.get("draw_id", f"draw_{i:06d}"))
        if key in references:
            raise ValueError(f"Duplicate evaluation draw ID {key}")
        refs = row.get("answer_aliases", [])
        pairs = row.get("qa_pairs", [])
        if len(refs) != 2 or not pairs:
            raise ValueError("Official ASQA dev evaluation requires two reference annotations and per-interpretation QA pairs")
        references[key] = {"annotations": [{"long_answer": ref} for ref in refs], "qa_pairs": pairs}
        predictions[key] = str(row.get("answer", ""))
    return references, predictions


def evaluate_longform(results, qa_predictions=None, scorer_seed=42):
    if qa_predictions is None:
        return {"n": len(results), "evaluation_status": "pending_official_qa_reader",
                "metric_protocol": "asqa_official", "note": "Run scripts/score_asqa.py after QA-reader inference; no proxy Disambig-F1 or DR is emitted."}
    if not results:
        raise ValueError("Cannot evaluate empty ASQA predictions")
    references, predictions = asqa_inputs(results)
    expected = {f"{key}_{i}" for key, ref in references.items() for i in range(len(ref["qa_pairs"]))}
    if set(qa_predictions) != expected:
        raise ValueError(f"QA-reader IDs mismatch: {len(expected - set(qa_predictions))} missing, {len(set(qa_predictions) - expected)} unexpected")
    official = official_asqa_module()
    previous = np.random.get_state()
    try:

        np.random.seed(scorer_seed)
        scores = official.compute_all_scores({"answers": predictions, "qa": qa_predictions}, references)
    except LookupError as exc:
        raise RuntimeError("Install NLTK sentence resources: python -m nltk.downloader punkt punkt_tab") from exc
    finally:
        np.random.set_state(previous)
    return {"n": len(results), "rouge_l": float(scores["rougeLsum"]) / 100,
            "disambig_f1": float(scores["QA-F1"]) / 100,
            "dr_score": float(scores["ovscore"]) / 100,
            "official_asqa_scores_percent": {key: float(value) for key, value in scores.items()},
            "metric_protocol": "asqa_official", "scorer_seed": scorer_seed,
            "evaluation_status": "complete"}


def dr_score(disambig, rouge):

    return math.sqrt(max(0.0, disambig * rouge))


def rouge_l_score(prediction, reference):

    import nltk
    from rouge_score import rouge_scorer
    def split(text):
        return "\n".join(nltk.sent_tokenize(text.lower()))
    return rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True).score(
        split(reference), split(prediction))["rougeLsum"].fmeasure


def rouge_l_max(prediction, references):
    return max((rouge_l_score(prediction, ref) for ref in references), default=0.0)


def evaluate_all_auto(results, qa_predictions=None):
    tasks = {row.get("task_type", "shortform") for row in results}
    if len(tasks) > 1:
        raise ValueError("Mixed short/long-form results must be scored separately")
    return evaluate_longform(results, qa_predictions) if tasks == {"longform"} else evaluate_all(results)


def evaluate_file(results_path, qa_predictions=None, verbose=False):
    payload = json.loads(Path(results_path).read_text())
    rows = payload["results"] if isinstance(payload, dict) else payload
    return evaluate_all_auto(rows, qa_predictions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results")
    p.add_argument("--qa-predictions")
    p.add_argument("--output")
    args = p.parse_args()
    qa = json.loads(Path(args.qa_predictions).read_text()) if args.qa_predictions else None
    metrics = evaluate_file(args.results, qa)
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
