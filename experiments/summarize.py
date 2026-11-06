import argparse
import json
from pathlib import Path

import numpy as np


def summarize(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    if not runs or any(r.get("run_status") != "complete" or r.get("metrics", {}).get("evaluation_status") != "complete" for r in runs):
        raise ValueError("All inputs must be complete and actually scored")
    for key in ["model", "method", "dataset", "code_sha256", "trace_position"]:
        if len({r.get(key) for r in runs}) != 1:
            raise ValueError(f"Do not average different {key} settings")
    if len({r["metrics"].get("metric_protocol") for r in runs}) != 1:
        raise ValueError("Metric protocols differ")
    if len({r.get("sampling_seed") for r in runs}) != len(runs):
        raise ValueError("Repeated sampling seeds would double-count one evaluation sample")
    def settings(run):
        cfg = json.loads(json.dumps(run["config"]))
        cfg["data"].pop("seed", None)
        return cfg
    if any(settings(r) != settings(runs[0]) for r in runs[1:]):
        raise ValueError("Settings other than question-sampling seed differ")
    keys = [key for key in ["f1", "em", "dr_score", "rouge_l", "disambig_f1"] if key in runs[0]["metrics"]]
    return {"n_runs": len(runs), "sampling_seeds": [r["sampling_seed"] for r in runs],
            "dataset": runs[0]["dataset"], "model": runs[0]["model"], "method": runs[0]["method"],
            "metric_protocol": runs[0]["metrics"]["metric_protocol"],
            "metrics": {key: float(np.mean([r["metrics"][key] for r in runs])) for key in keys},
            "aggregation": "arithmetic mean of per-sample benchmark scores; DR computed before this average"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="+")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = summarize(args.results)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
