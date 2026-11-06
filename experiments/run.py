import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from tqdm import tqdm

from data.loader import load_dataset, get_docs
from experiments.evaluate import evaluate_all_auto
from src.gbtrag import GBTRAG
from src.llm import LLM, Embedder


def _config_path(path):
    value = Path(path).expanduser()
    return value if value.is_absolute() or value.exists() else ROOT / value


def _load_config(config_path="configs/default.yaml", overrides=(), dataset=None, server_config=None):
    cfg = OmegaConf.load(ROOT / "configs/default.yaml")
    requested = _config_path(config_path)
    if not requested.is_file():
        raise FileNotFoundError(f"Config not found: {requested}")
    if requested.resolve() != (ROOT / "configs/default.yaml").resolve():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(requested))
    name = dataset or cfg.data.name
    dataset_file = ROOT / "configs" / f"{name}.yaml"
    if dataset_file.is_file():

        cfg = OmegaConf.merge(OmegaConf.load(ROOT / "configs/default.yaml"),
                              OmegaConf.load(dataset_file),
                              OmegaConf.load(requested) if requested.resolve() != (ROOT / "configs/default.yaml").resolve() else {})
    if dataset:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(ROOT / "configs" / f"{dataset}.yaml"))
    if server_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(_config_path(server_config)))
    for override in overrides:
        if "=" not in override or override.startswith("--"):
            raise ValueError(f"Invalid override {override!r}; use key=value")
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))


def _apply_local_test(cfg):
    if cfg.local_test.enabled:
        cfg.data.n_samples = int(cfg.local_test.n_samples)
    return cfg


def _code_hash():
    digest = hashlib.sha256()
    for folder in ["src", "experiments", "data"]:
        for path in sorted((ROOT / folder).glob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def run_method(method_name, method, dataset, cfg, result_dir, output_suffix="", resume=None):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = [{"draw_id": s["draw_id"], "source_id": s["id"], "question": s["question"]} for s in dataset]
    payload_config = OmegaConf.to_container(cfg, resolve=True)
    dataset_digest = hashlib.sha256(json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    index_metadata = Path(str(cfg.retrieval.index_dir)) / "metadata.json"
    index_digest = hashlib.sha256(index_metadata.read_bytes()).hexdigest() if cfg.retrieval.scope == "global" and index_metadata.is_file() else None
    identity = {"method": method_name, "config": payload_config,
                "dataset_sha256": dataset_digest, "index_metadata_sha256": index_digest,
                "sampling_manifest": manifest, "code_sha256": _code_hash()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if resume:
        save_path = Path(resume)
        previous = json.loads(save_path.read_text())
        if previous.get("run_fingerprint") != fingerprint:
            raise ValueError("Resume configuration/code/sampling mismatch")
        if previous.get("postprocess_only"):
            raise ValueError("Cannot resume a round-postprocessing artifact as a generation run")
        if previous.get("run_status") == "complete":
            print("Generation already complete:", save_path)
            return previous["metrics"]
        results = previous.get("results", [])
        if any(r.get("error") for r in results):
            raise ValueError("Resume file contains errors; remove the failed row only after fixing its cause")
    else:
        suffix = f"_{output_suffix}" if output_suffix else ""
        save_path = result_dir / f"{method_name}_{cfg.data.name}_seed{cfg.data.seed}_{time.time_ns()}{suffix}.json"
        results = []
    if len(results) > len(dataset):
        raise ValueError("Resume has more rows than the current dataset")
    for i, row in enumerate(results):
        if row.get("draw_id") != dataset[i]["draw_id"] or row.get("id") != dataset[i]["id"]:
            raise ValueError("Resume row order differs from sampling manifest")
    base = {**identity, "run_fingerprint": fingerprint, "model": str(cfg.llm.model),
            "dataset": str(cfg.data.name), "sampling_seed": int(cfg.data.seed),
            "sampling_with_replacement": bool(cfg.data.sample_with_replacement),
            "expected_n_samples": len(dataset), "results": results,
            "run_status": "running", "metrics": {"evaluation_status": "pending_generation"}}
    _json_atomic(save_path, base)
    print("Results:", save_path)
    for i in tqdm(range(len(results), len(dataset)), desc=method_name):
        sample = dataset[i]
        try:

            result = method.run(sample["question"], get_docs(sample),
                                task_type=sample.get("task_type", "shortform"))
            if method_name != "gbtrag" and sample.get("task_type") == "longform" and cfg.gbtrag.compress_output:
                from src.prompts import COMPRESSION_PROMPT, fmt
                raw = result["answer"]
                result["a_final_raw"] = raw
                result["answer"] = method.llm(fmt(COMPRESSION_PROMPT, A_final=raw,
                    target_words=cfg.gbtrag.compress_target_words,
                    structure_hint=cfg.gbtrag.compress_structure_hint), max_tokens=cfg.llm.final_max_tokens)
                result["answer_stage"] = "post_compression"
            if sample.get("task_type", "shortform") == "shortform":
                result["answer"] = " ".join(result.get("answer", "").split()[:10])
        except KeyboardInterrupt:
            base["run_status"] = "interrupted"
            base["interrupted_at_draw"] = i
            base["metrics"] = {"evaluation_status": "incomplete_generation"}
            _json_atomic(save_path, base)
            print(f"Interrupted; completed samples preserved at {save_path}")
            raise
        except Exception as exc:
            result = {"answer": "", "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()

        result.update({"id": sample["id"], "draw_id": sample["draw_id"],
                       "draw_index": sample["draw_index"], "question": sample["question"],
                       "gold": sample["answer"], "answer_aliases": sample["answer_aliases"],
                       "qa_pairs": sample.get("qa_pairs", []),
                       "short_answers": sample.get("short_answers", []),
                       "task_type": sample.get("task_type", "shortform"),
                       "dataset": sample.get("dataset", cfg.data.name)})
        results.append(result)
        _json_atomic(save_path, base)
        if result.get("error") and not cfg.experiment.keep_going:
            base["run_status"] = "failed"
            base["metrics"] = {"evaluation_status": "incomplete_generation"}
            _json_atomic(save_path, base)
            raise RuntimeError(f"Sample failed; partial output saved at {save_path}")
    errors = sum(bool(r.get("error")) for r in results)
    metrics = evaluate_all_auto(results) if not errors else {"evaluation_status": "incomplete_generation"}
    base.update({"run_status": "complete" if not errors else "completed_with_errors",
                 "n_samples": len(results), "n_errors": errors, "metrics": metrics})
    _json_atomic(save_path, base)
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--server-config")
    parser.add_argument("--dataset", choices=["hotpotqa", "musique", "asqa", "twowiki"])
    parser.add_argument("--model", choices=["llama8b", "qwen7b"])
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--methods")
    parser.add_argument("--port", type=int)
    parser.add_argument("--max_rounds", type=int)
    parser.add_argument("--coverage_threshold", type=float, help="Coverage margin tau (not epsilon)")
    parser.add_argument("--convergence_epsilon", type=float)
    parser.add_argument("--n_subaspects", type=int)
    parser.add_argument("--ablation", choices=["no_residual", "no_draft_critique", "no_coverage_loss", "word_overlap_coverage", "no_dim_discovery", "additive_compose", "llm_tree", "no_compress", "single_round", "no_monotone_gate", "no_dead_leaf_pruning"])
    parser.add_argument("--no-cache", "--no_cache", action="store_true")
    parser.add_argument("--no-compress", "--no_compress", action="store_true")
    parser.add_argument("--output_suffix", default="")
    parser.add_argument("--resume", help="Resume one saved result JSON with identical code/config/sample manifest")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and question sampling without loading models")
    args, overrides = parser.parse_known_args()
    cfg = _load_config(args.config, (), dataset=args.dataset, server_config=args.server_config)
    if args.model:
        cfg = OmegaConf.merge(cfg, cfg.model_presets[args.model])
    explicit = {}
    for value, key in [(args.samples, "data.n_samples"), (args.seed, "data.seed"),
                       (args.max_rounds, "gbtrag.max_rounds"),
                       (args.coverage_threshold, "gbtrag.coverage_threshold"),
                       (args.convergence_epsilon, "gbtrag.convergence_epsilon"),
                       (args.n_subaspects, "gbtrag.sub_aspects_per_dim")]:
        if value is not None:
            explicit[key] = value
    if args.methods:
        explicit["experiment.methods"] = [x.strip() for x in args.methods.split(",")]
    if args.port:
        explicit["llm.base_url"] = f"http://127.0.0.1:{args.port}/v1"
    if args.no_cache:
        explicit["experiment.cache_dir"] = None
    if args.no_compress:
        explicit["gbtrag.compress_output"] = False
    ablations = {"no_residual": ("no_residual", True), "no_draft_critique": ("phase0_mode", "decompose"),
                 "word_overlap_coverage": ("coverage_fn", "word_overlap"), "no_dim_discovery": ("enable_dimension_discovery", False),
                 "additive_compose": ("compose_op", "additive"), "llm_tree": ("tree_mode", "llm"),
                 "no_compress": ("compress_output", False), "single_round": ("single_round", True),
                 "no_monotone_gate": ("enable_monotone_gate", False),
                 "no_dead_leaf_pruning": ("enable_dead_leaf_pruning", False)}
    if args.ablation:
        if args.ablation == "no_coverage_loss":
            parser.error("The paper replaces embedding coverage with text-only critique; "
                         "its historical implementation was not supplied. "
                         "Use --ablation word_overlap_coverage only for the separately named "
                         "word-overlap diagnostic, not for the paper's critique ablation.")
        key, value = ablations[args.ablation]
        explicit["gbtrag." + key] = value
    for key, value in explicit.items():
        OmegaConf.update(cfg, key, value)
    for value in overrides:
        if "=" not in value or value.startswith("--"):
            parser.error(f"Unknown option {value!r}; use key=value for config overrides")
    cfg = _apply_local_test(OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides)))
    for section, key in [("embedding", "model_path"), ("reranker", "model_path"),
                         ("retrieval", "index_dir"), ("data", "path"),
                         ("experiment", "output_dir"), ("experiment", "cache_dir")]:
        value = getattr(cfg[section], key)
        if value and not Path(value).is_absolute():
            cfg[section][key] = str(ROOT / value)
    if cfg.gbtrag.max_rounds < 0 or not (0 < cfg.gbtrag.coverage_threshold < 1):
        parser.error("Require max_rounds>=0 and 0<coverage_threshold<1")
    dataset = load_dataset(cfg)
    if not dataset:
        raise RuntimeError("Dataset is empty")
    print(OmegaConf.to_yaml(cfg))
    print(f"Selected {len(dataset)} draws; unique questions={len({s['id'] for s in dataset})}")
    if args.dry_run:
        print("DRY RUN: no model loaded, no predictions generated")
        return
    llm = LLM(cfg.llm.base_url, cfg.llm.model, cfg.llm.temperature,
              cfg.llm.max_tokens, cfg.llm.timeout, cfg.experiment.cache_dir,
              cfg.llm.seed, cfg.llm.max_retries)
    embedder = Embedder(**dict(cfg.embedding))
    if args.resume and len(cfg.experiment.methods) != 1:
        parser.error("--resume requires exactly one method")
    for name in cfg.experiment.methods:
        if name == "gbtrag":
            method = GBTRAG(cfg, llm=llm, embedder=embedder)
        else:
            from src.baselines import get_method
            print("NOTE: bundled baseline is a simplified reference, not an official paper baseline reproduction")
            method = get_method(name, llm, embedder, cfg)
        run_method(name, method, dataset, cfg, Path(cfg.experiment.output_dir) / cfg.data.name,
                   args.output_suffix, args.resume)


if __name__ == "__main__":
    main()
