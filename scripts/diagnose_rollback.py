import argparse
import json
from pathlib import Path


def analyze(results_file):
    payload = json.loads(Path(results_file).read_text())
    samples = payload["results"] if isinstance(payload, dict) else payload
    logs = [log for sample in samples for log in sample.get("round_logs", [])
            if log.get("decision") in {"accept", "rollback"}]
    violations = [log for log in logs if log.get("gate_enabled", True) and
                  log["coverage_loss_after"] > log["coverage_loss_before"]]
    report = {"n_samples": len(samples), "attempts": len(logs),
              "accepted": sum(log["decision"] == "accept" for log in logs),
              "rolled_back": sum(log["decision"] == "rollback" for log in logs),
              "strict_gate_violations": len(violations),
              "empty_residual_evidence": sum(log.get("residual_evidence_empty", False) for log in logs),
              "zero_update_questions": sum(row.get("num_boosting_rounds", 0) == 0 for row in samples),
              "note": "Loss values from different coordinate systems are not a fixed-objective convergence curve."}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    report = analyze(parser.parse_args().results)
    raise SystemExit(1 if report["strict_gate_violations"] else 0)
