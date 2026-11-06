import json
import os
import random
import sys


def _join_context_hotpot(raw_context: list) -> list:

    paragraphs = []
    for entry in raw_context:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            print(f"[loader] Warning: unexpected context entry format: {entry!r}")
            continue
        title = entry[0] or ""
        sentences = entry[1]
        if isinstance(sentences, list):
            text = " ".join(s for s in sentences if isinstance(s, str))
        elif isinstance(sentences, str):

            text = sentences
        else:
            text = ""
        text = " ".join(text.split())
        paragraphs.append({"title": str(title), "text": text})
    return paragraphs


def _sample_list(items: list, n_samples: int, seed: int, replacement: bool = False) -> list:

    if n_samples <= 0:
        return list(items)
    if not items:
        raise ValueError("Cannot sample from an empty dataset")
    rng = random.Random(seed)
    if replacement:
        return rng.choices(items, k=n_samples)
    if n_samples > len(items):
        raise ValueError(f"Requested {n_samples} questions from pool {len(items)} without replacement")
    return rng.sample(items, n_samples)


def _get_n_samples(cfg) -> int:

    local_test = getattr(cfg, "local_test", None)
    if local_test is not None and getattr(local_test, "enabled", False):
        return int(local_test.n_samples)
    return int(getattr(cfg.data, "n_samples", -1))


def _safe_open_json(path: str, dataset_name: str):

    if not os.path.exists(path):
        print(
            f"[loader] ERROR: {dataset_name} file not found at: {path}\n"
            f"         Please download the dataset and place it at the path\n"
            f"         shown above. See scripts/download_data.sh for commands."
        )
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(
            f"[loader] ERROR: Failed to parse {dataset_name} JSON file.\n"
            f"         Path : {path}\n"
            f"         Error: {e}"
        )
        return None


def _safe_open_jsonl(path: str, dataset_name: str) -> list:

    if not os.path.exists(path):
        print(
            f"[loader] ERROR: {dataset_name} file not found at: {path}\n"
            f"         Please download the dataset and place it at the path\n"
            f"         shown above. See scripts/download_data.sh for commands."
        )
        return None
    records = []
    parse_errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                parse_errors += 1
                if parse_errors <= 3:
                    print(
                        f"[loader] Warning: JSON parse error on line {line_no} "
                        f"of {path}: {e}"
                    )
    if parse_errors > 3:
        print(f"[loader] Warning: {parse_errors} total parse errors in {path}.")
    return records


def load_hotpotqa(path: str, n_samples: int = -1, seed: int = 42) -> list:

    raw = _safe_open_json(path, "HotPotQA")
    if raw is None:
        return []

    if not isinstance(raw, list):
        print(
            f"[loader] ERROR: Expected a list at the top level of the "
            f"HotPotQA file, got {type(raw).__name__}."
        )
        return []

    samples = []
    skipped = 0
    for raw_sample in raw:
        try:
            context = _join_context_hotpot(raw_sample.get("context", []))
            answer = str(raw_sample.get("answer", ""))
            sample = {
                "id":              str(raw_sample.get("_id", "")),
                "question":        str(raw_sample.get("question", "")),
                "answer":          answer,
                "answer_aliases":  [answer],
                "context":         context,
                "type":            str(raw_sample.get("type", "")),
                "supporting_facts": raw_sample.get("supporting_facts", []),
                "dataset":         "hotpotqa",
            }
            samples.append(sample)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"[loader] Warning: skipping malformed HotPotQA sample: {e}")

    if skipped:
        print(f"[loader] HotPotQA: skipped {skipped} malformed samples.")

    return _sample_list(samples, n_samples, seed)


def load_musique(path: str, n_samples: int = -1, seed: int = 42) -> list:

    raw = _safe_open_jsonl(path, "MuSiQue")
    if raw is None:
        return []

    samples = []
    skipped = 0
    unanswerable = 0
    for raw_sample in raw:

        if not raw_sample.get("answerable", True):
            unanswerable += 1
            continue

        try:

            context = []
            for p in raw_sample.get("paragraphs", []):
                title = str(p.get("title", ""))
                text = " ".join(str(p.get("paragraph_text", "")).split())
                context.append({"title": title, "text": text})

            answer = str(raw_sample.get("answer", ""))
            raw_aliases = raw_sample.get("answer_aliases", [])

            aliases = list(raw_aliases) if raw_aliases else []
            if answer and answer not in aliases:
                aliases.insert(0, answer)


            decomp = raw_sample.get("question_decomposition", [])
            supporting_facts = []
            for step in decomp:
                para_idx = step.get("paragraph_support_idx")
                if para_idx is not None and para_idx < len(context):
                    supporting_facts.append(
                        [context[para_idx]["title"], para_idx]
                    )

            sample = {
                "id":              str(raw_sample.get("id", "")),
                "question":        str(raw_sample.get("question", "")),
                "answer":          answer,
                "answer_aliases":  aliases,
                "context":         context,
                "type":            "multihop",
                "supporting_facts": supporting_facts,
                "dataset":         "musique",
            }
            samples.append(sample)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"[loader] Warning: skipping malformed MuSiQue sample: {e}")

    if unanswerable:
        print(f"[loader] MuSiQue: filtered out {unanswerable} unanswerable samples.")
    if skipped:
        print(f"[loader] MuSiQue: skipped {skipped} malformed samples.")

    return _sample_list(samples, n_samples, seed)


def load_twowiki(path: str, n_samples: int = -1, seed: int = 42) -> list:

    raw = _safe_open_json(path, "2WikiMultiHopQA")
    if raw is None:
        return []

    if not isinstance(raw, list):
        print(
            f"[loader] ERROR: Expected a list at the top level of the "
            f"2WikiMultiHopQA file, got {type(raw).__name__}."
        )
        return []

    samples = []
    skipped = 0
    for raw_sample in raw:
        try:
            context = _join_context_hotpot(raw_sample.get("context", []))
            answer = str(raw_sample.get("answer", ""))
            sample = {
                "id":              str(raw_sample.get("_id", "")),
                "question":        str(raw_sample.get("question", "")),
                "answer":          answer,
                "answer_aliases":  [answer],
                "context":         context,
                "type":            str(raw_sample.get("type", "")),
                "supporting_facts": raw_sample.get("supporting_facts", []),
                "dataset":         "twowiki",
            }
            samples.append(sample)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"[loader] Warning: skipping malformed 2WikiMultiHopQA sample: {e}")

    if skipped:
        print(f"[loader] 2WikiMultiHopQA: skipped {skipped} malformed samples.")

    return _sample_list(samples, n_samples, seed)


def load_asqa(path: str, n_samples: int = -1, seed: int = 42,
              replacement: bool = False) -> list:

    raw = _safe_open_json(path, "ASQA")
    if raw is None:
        raise ValueError(f"Could not load ASQA: {path}")
    if isinstance(raw, dict):
        raw = raw.get("dev", raw.get("data", raw))
    if isinstance(raw, dict):
        entries = [(str(key), sample) for key, sample in raw.items()]
    elif isinstance(raw, list):
        entries = [(str(i), sample) for i, sample in enumerate(raw)]
    else:
        raise ValueError("ASQA must be an example list or ID-to-example mapping")
    samples = []
    for default_id, sample in entries:
        if not isinstance(sample, dict):
            raise ValueError(f"Malformed ASQA example {default_id}")
        question = sample.get("ambiguous_question", sample.get("question", ""))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing ASQA question {default_id}")
        annotations = sample.get("annotations", [])
        references = [a["long_answer"] for a in annotations if isinstance(a, dict) and a.get("long_answer")]
        if not references:
            references = sample.get("answers", [])
        if not references and sample.get("answer"):
            references = [sample["answer"]]
        if not references or not all(isinstance(r, str) for r in references):
            raise ValueError(f"Missing ASQA reference answers {default_id}")
        qa_pairs = sample.get("qa_pairs", [])
        if not isinstance(qa_pairs, list) or not qa_pairs:
            raise ValueError(f"Missing per-interpretation QA pairs {default_id}")
        for qa in qa_pairs:
            if not isinstance(qa, dict) or not qa.get("question") or not qa.get("short_answers"):
                raise ValueError(f"Malformed ASQA QA pair {default_id}")


        docs = []
        for doc in sample.get("docs", []):
            if not isinstance(doc, dict):
                raise ValueError(f"Malformed retrieval document {default_id}")
            text = str(doc.get("text", doc.get("paragraph_text", ""))).strip()
            if text:
                docs.append({"title": str(doc.get("title", "")), "text": text})
        samples.append({"id": str(sample.get("sample_id", sample.get("id", default_id))),
                        "question": question, "answer": references[0],
                        "answer_aliases": references,
                        "short_answers": [a for qa in qa_pairs for a in qa["short_answers"]],
                        "qa_pairs": qa_pairs, "context": docs, "dataset": "asqa",
                        "task_type": "longform"})
    return _sample_list(samples, n_samples, seed, replacement)


def load_dataset(cfg) -> list:

    name    = str(cfg.data.name).lower()
    path    = str(cfg.data.path)
    seed    = int(getattr(cfg.data, "seed", 42))
    n       = _get_n_samples(cfg)

    loaders = {
        "hotpotqa": load_hotpotqa,
        "musique":  load_musique,
        "twowiki":  load_twowiki,
        "asqa":     load_asqa,
    }

    if name not in loaders:
        print(
            f"[loader] ERROR: Unknown dataset name '{name}'. "
            f"Supported: {list(loaders.keys())}"
        )
        return []

    samples = loaders[name](path, n_samples=-1, seed=seed)


    validated = []
    for i, s in enumerate(samples):
        try:
            assert isinstance(s["context"], list), \
                f"context is {type(s['context']).__name__}, expected list"
            for p in s["context"]:
                assert isinstance(p, dict), \
                    f"context entry is {type(p).__name__}, expected dict"
                assert isinstance(p.get("title"), str), \
                    f"'title' is {type(p.get('title')).__name__}, expected str"
                assert isinstance(p.get("text"), str), \
                    f"'text' is {type(p.get('text')).__name__}, expected str"
            validated.append(s)
        except (AssertionError, KeyError, AttributeError) as e:
            print(f"  [WARN] Skipping malformed sample {i}: {e}")

    n_dropped = len(samples) - len(validated)
    if n_dropped:
        print(f"[loader] Validation: dropped {n_dropped} malformed sample(s) "
              f"from {name}.")

    if validated:
        print(f"[loader] Loaded {len(validated)} samples from {name} ({path})")
    else:
        print(
            f"[loader] WARNING: No samples loaded from {name}. "
            f"Check that the file exists at '{path}'."
        )

    expected = getattr(cfg.data, "expected_pool_size", None)
    if expected is not None and len(validated) != int(expected):
        raise ValueError(f"Expected {expected} {name} questions but loaded {len(validated)}; check the data file")
    replacement = bool(getattr(cfg.data, "sample_with_replacement", False))
    selected = _sample_list(validated, n, seed, replacement)
    if str(getattr(cfg.retrieval, "scope", "global")) == "candidate_debug" and any(not s["context"] for s in selected):
        raise ValueError("candidate_debug requires independently retrieved docs; gold QA contexts are not a retrieval corpus")
    return [{**sample, "draw_id": f"draw_{i:06d}", "draw_index": i,
             "sampling_seed": seed, "pool_size": len(validated),
             "sampling_with_replacement": replacement} for i, sample in enumerate(selected)]


def get_docs(sample: dict) -> list:

    return sample.get("context", [])


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description="Test BRIGHT dataset loaders")
    parser.add_argument("--hotpotqa", default="./data/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--musique",  default="./data/musique/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--twowiki",  default="./data/twowiki/dev.json")
    parser.add_argument("--n",        type=int, default=3, help="Samples to show")
    args = parser.parse_args()

    def show_dataset(name: str, samples: list, n: int = 2):
        if not samples:
            print(f"  (no samples loaded)\n")
            return
        print(f"  Total samples : {len(samples)}")
        for i, s in enumerate(samples[:n]):
            print(f"\n  --- Sample {i} ---")
            print(f"  id              : {s['id']}")
            print(f"  question        : {s['question']}")
            print(f"  answer          : {s['answer']}")
            print(f"  answer_aliases  : {s['answer_aliases']}")
            print(f"  type            : {s['type']}")
            print(f"  dataset         : {s['dataset']}")
            print(f"  num paragraphs  : {len(s['context'])}")
            if s["context"]:
                first = s["context"][0]
                print(f"  first para title: {first['title']}")
                print(f"  first para text : {first['text'][:120]}...")
            print(f"  supporting_facts: {s['supporting_facts'][:3]}")

    print("=" * 60)
    print("HotPotQA")
    print("=" * 60)
    hpqa = load_hotpotqa(args.hotpotqa, n_samples=args.n, seed=42)
    show_dataset("hotpotqa", hpqa)

    print("\n" + "=" * 60)
    print("MuSiQue")
    print("=" * 60)
    musi = load_musique(args.musique, n_samples=args.n, seed=42)
    show_dataset("musique", musi)

    print("\n" + "=" * 60)
    print("2WikiMultiHopQA")
    print("=" * 60)
    wiki = load_twowiki(args.twowiki, n_samples=args.n, seed=42)
    show_dataset("twowiki", wiki)


    print("\n" + "=" * 60)
    print("get_docs() sanity check")
    print("=" * 60)
    for name, samples in [("hotpotqa", hpqa), ("musique", musi), ("twowiki", wiki)]:
        if samples:
            docs = get_docs(samples[0])
            has_title = all("title" in d for d in docs)
            has_text  = all("text"  in d for d in docs)
            print(
                f"  {name}: {len(docs)} docs, "
                f"all have 'title'={has_title}, 'text'={has_text}"
            )
