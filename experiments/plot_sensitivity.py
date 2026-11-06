import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


SWEEPS = [
    {
        "key":      "A",
        "title":    "Sweep A — Convergence epsilon (ε)",
        "xlabel":   "ε  (convergence_epsilon)",
        "suffix_prefix": "sweep_a_",
        "values":   [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20],
        "default":  0.01,
        "xscale":   "log",
        "xfmt":     lambda v: str(v),
    },
    {
        "key":      "B",
        "title":    "Sweep B — Max boosting rounds (M)",
        "xlabel":   "M  (max_rounds)",
        "suffix_prefix": "sweep_b_",
        "values":   [1, 2, 3, 4, 5, 7, 10],
        "default":  3,
        "xscale":   "linear",
        "xfmt":     lambda v: str(int(v)),
    },
    {
        "key": "C", "title": "Sweep C — Sub-aspects per generic dimension (S)",
        "xlabel": "S (sub_aspects_per_dim; generic decomposition only)",
        "suffix_prefix": "sweep_c_", "values": [1, 2, 3, 4, 5],
        "default": 3, "xscale": "linear", "xfmt": lambda v: str(int(v)),
    },
    {
        "key":      "D",
        "title":    "Sweep D — Coverage threshold (τ)",
        "xlabel":   "τ  (coverage_threshold)",
        "suffix_prefix": "sweep_d_",
        "values":   [0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        "default":  0.65,
        "xscale":   "linear",
        "xfmt":     lambda v: f"{v:.2f}",
    },
]

METRICS = [
    ("dr_score",      "DR",    "#1f77b4",  "o-",  True),
    ("rouge_l",       "RL",    "#ff7f0e",  "s--", True),
    ("disambig_f1",   "DF1",   "#2ca02c",  "^-",  True),
]

DIAG_METRICS = [
    ("diag_rollback_rate",   "Rollback %",  "#9467bd", "D:",  False),
    ("diag_avg_num_rounds",  "Avg rounds",  "#8c564b", "x--", False),
]


def load_latest(results_dir: str, suffix: str):
    pattern = os.path.join(results_dir, f"gbtrag_*_{suffix}.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def extract_metrics(data):
    if data is None:
        return None
    m = data.get("metrics", {})
    return {
        "dr_score":            m.get("dr_score",           None),
        "rouge_l":             m.get("rouge_l",            None),
        "disambig_f1":         m.get("disambig_f1",        None),
        "rouge_l_std":         m.get("rouge_l_std",        None),
        "disambig_f1_std":     m.get("disambig_f1_std",    None),
        "diag_rollback_rate":  m.get("diag_rollback_rate", None),
        "diag_avg_num_rounds": m.get("diag_avg_num_rounds",None),
        "n_errors":            m.get("n_errors",           0),
    }


def value_to_suffix(sweep, v):

    fmt = sweep["xfmt"](v)
    return f"{sweep['suffix_prefix']}{fmt}"


def plot_sensitivity(results_dir: str, figures_dir: str, baseline_suffix: str):
    os.makedirs(figures_dir, exist_ok=True)

    baseline_data = load_latest(results_dir, baseline_suffix)
    baseline_m    = extract_metrics(baseline_data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, sweep in zip(axes, SWEEPS):
        xs_all   = []
        ys       = {mk: [] for mk, _, _, _, _ in METRICS + DIAG_METRICS}
        missing  = []

        for v in sweep["values"]:
            suffix = value_to_suffix(sweep, v)
            data   = load_latest(results_dir, suffix)
            m      = extract_metrics(data)

            if m is None or m.get("dr_score") is None:
                missing.append(v)

                xs_all.append(v)
                for mk, *_ in METRICS + DIAG_METRICS:
                    ys[mk].append(float("nan"))
            else:
                xs_all.append(v)
                for mk, *_ in METRICS + DIAG_METRICS:
                    ys[mk].append(m.get(mk) if m.get(mk) is not None else float("nan"))

        xs = np.array(xs_all, dtype=float)


        ax2 = ax.twinx()

        for mk, label, color, style, primary in METRICS + DIAG_METRICS:
            yv = np.array(ys[mk], dtype=float)
            if primary:
                ax.plot(xs, yv, style, color=color, label=label,
                        linewidth=1.8, markersize=6, markerfacecolor="white",
                        markeredgewidth=1.5)
            else:
                ax2.plot(xs, yv, style, color=color, label=label,
                         linewidth=1.2, markersize=5, alpha=0.7)


        ax.axvline(sweep["default"], color="gray", linestyle=":", linewidth=1.2,
                   label=f"default ({sweep['default']})")


        if baseline_m:
            for mk, label, color, _, primary in METRICS:
                bv = baseline_m.get(mk)
                if bv is not None and primary:
                    ax.axhline(bv, color=color, linestyle="--", linewidth=0.8,
                               alpha=0.4)


        ax.set_xscale(sweep["xscale"])
        ax.set_xlabel(sweep["xlabel"], fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(sweep["title"], fontsize=11, fontweight="bold")
        ax.set_ylim(0, 0.65)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(True, which="both", alpha=0.3, linewidth=0.6)

        ax2.set_ylabel("Rollback % / Avg rounds", fontsize=9, color="gray")
        ax2.set_ylim(0, 5)
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)


        ax.set_xticks(sweep["values"])
        ax.set_xticklabels(
            [sweep["xfmt"](v) for v in sweep["values"]],
            fontsize=8, rotation=30, ha="right"
        )


        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=8, loc="lower right", framealpha=0.85)


        if missing:
            ax.text(0.02, 0.02, f"missing: {missing}",
                    transform=ax.transAxes, fontsize=7,
                    color="red", va="bottom")


    fig.suptitle(
        "GBTRAG — Parameter Sensitivity Analysis (ASQA, n=50)\n"
        "Dashed horizontal lines = baseline (ε=0.01, M=5, τ=0.70, ct=0.08)",
        fontsize=12, y=1.01,
    )

    plt.tight_layout()

    out_png = os.path.join(figures_dir, "sensitivity_analysis.png")
    out_pdf = os.path.join(figures_dir, "sensitivity_analysis.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot GBTRAG sensitivity sweep results")
    parser.add_argument("--results_dir",     default="./results/asqa")
    parser.add_argument("--figures_dir",     default="./figures")
    parser.add_argument("--baseline_suffix", default="diag_no_fixb")
    args = parser.parse_args()
    plot_sensitivity(args.results_dir, args.figures_dir, args.baseline_suffix)


if __name__ == "__main__":
    main()
