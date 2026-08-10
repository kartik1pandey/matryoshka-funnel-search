"""Generate the two result charts embedded in the public README.

Reads the real evaluation/cost-projection reports (not hand-typed numbers)
so the charts stay in sync with docs/08_results.md by construction. Colors
are the validated categorical palette from the project's dataviz skill
(references/palette.md) — blue for the Matryoshka model / funnel search,
orange for the baseline / plain brute-force search, kept consistent across
both charts since they mean the same thing in each.

Usage: pip install -e ".[viz]"; python scripts/generate_result_charts.py
Reads: checkpoints/{evaluation,cost_projection}_report.json
Writes: assets/recall_by_dimension.png, assets/latency_comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

CHECKPOINT_DIR = Path("checkpoints")
ASSETS_DIR = Path("assets")

BLUE = "#2a78d6"  # Matryoshka model / funnel search
ORANGE = "#eb6834"  # baseline model / plain brute-force search
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "text.color": INK_PRIMARY,
        "axes.edgecolor": AXIS_LINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def _style_axes(ax, y_gridlines: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_LINE)
    if y_gridlines:
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def plot_recall_by_dimension(report: dict) -> None:
    dims = sorted(int(d) for d in report["matryoshka_quality_by_dim"])
    mrl = [report["matryoshka_quality_by_dim"][str(d)]["recall@10"] for d in dims]
    baseline = [report["baseline_quality_by_dim"][str(d)]["recall@10"] for d in dims]
    x = list(range(len(dims)))  # evenly spaced positions for the log2-scale dims

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _style_axes(ax)

    ax.plot(x, mrl, color=BLUE, linewidth=2, marker="o", markersize=8, label="Matryoshka", zorder=3)
    ax.plot(
        x, baseline, color=ORANGE, linewidth=2, marker="o", markersize=8, label="Baseline", zorder=3
    )

    for i in (0, len(dims) - 1):
        ax.annotate(
            f"{mrl[i]:.3f}",
            (x[i], mrl[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color=BLUE,
            fontweight="bold",
        )
        ax.annotate(
            f"{baseline[i]:.3f}",
            (x[i], baseline[i]),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=9,
            color=ORANGE,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Embedding dimension (truncated prefix length)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, max(mrl) * 1.2)
    ax.legend(frameon=False, loc="upper left", fontsize=11)

    fig.suptitle(
        "Recall@10 by Matryoshka dimension",
        x=0.09,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_title(
        "1,500 held-out validation queries · full 15,000-item ABO catalog",
        loc="left",
        fontsize=10,
        color=INK_SECONDARY,
        pad=12,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS_DIR / "recall_by_dimension.png")
    plt.close(fig)


def plot_latency_comparison(report: dict) -> dict:
    """Returns the plotted numbers (ms) so `main()` can print a copy-pasteable
    summary — the exact prevention for the bug this replaced: this script and
    scripts/evaluate.py both independently wall-clock-benchmark "brute-force
    full-dim vs. funnel search," and two independent benchmark runs on a
    shared dev machine do NOT agree to the millisecond (20.1ms vs. 26.0ms was
    observed for the *same* operation across two runs — ordinary noise, not a
    bug in either measurement). `cost_projection_report.json` (this report)
    is the sole source for any latency number quoted publicly (README,
    docs/08_results.md's Cost Projection section) — scripts/evaluate.py's own
    latency numbers exist only to compare Matryoshka-vs-baseline latency
    (docs/08_results.md's Week 3 Latency section), never to be re-quoted here
    or in the README as if they were the same measurement.
    """
    measured = report["measured_latency_seconds"]
    extrapolated = report["extrapolated_latency_seconds_at_full_catalog"]
    n_measured = report["n_measured"]
    n_full = report["full_catalog_size"]
    low_dim = report["funnel_low_dim"]

    funnel_ms = [
        (measured["funnel_stage1"] + measured["funnel_stage2"]) * 1000,
        extrapolated["funnel_search"] * 1000,
    ]
    brute_force_ms = [
        measured["brute_force_full_dim"] * 1000,
        extrapolated["brute_force_full_dim"] * 1000,
    ]
    group_labels = [f"{n_measured:,} items\n(measured)", f"{n_full:,} items\n(extrapolated)"]
    funnel_label = f"Funnel search (Stage 1 @ {low_dim}-dim)"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _style_axes(ax)

    x = [0, 1]
    width = 0.32
    bars_funnel = ax.bar(
        [xi - width / 2 for xi in x], funnel_ms, width, color=BLUE, label=funnel_label, zorder=3
    )
    bars_brute = ax.bar(
        [xi + width / 2 for xi in x],
        brute_force_ms,
        width,
        color=ORANGE,
        label="Brute-force, full 512-dim",
        zorder=3,
    )

    for bars in (bars_funnel, bars_brute):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}ms",
                (bar.get_x() + bar.get_width() / 2, height),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=10,
                fontweight="bold",
                color=INK_PRIMARY,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Mean query latency (ms)")
    ax.set_ylim(0, max(brute_force_ms) * 1.18)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.legend(frameon=False, loc="upper left", fontsize=11)

    fig.suptitle(
        "Funnel search vs. brute force: query latency",
        x=0.09,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_title(
        "Measured on the real 15,000-item catalog (Matryoshka embeddings)\n"
        "Extrapolated to the full 147,702-item ABO catalog",
        loc="left",
        fontsize=10,
        color=INK_SECONDARY,
        pad=12,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS_DIR / "latency_comparison.png")
    plt.close(fig)

    return {
        "low_dim": low_dim,
        "measured_funnel_ms": funnel_ms[0],
        "measured_brute_force_ms": brute_force_ms[0],
        "extrapolated_funnel_ms": funnel_ms[1],
        "extrapolated_brute_force_ms": brute_force_ms[1],
        "n_measured": n_measured,
        "n_full": n_full,
    }


def main() -> None:
    evaluation_report = json.loads(
        (CHECKPOINT_DIR / "evaluation_report.json").read_text(encoding="utf-8")
    )
    cost_projection_report = json.loads(
        (CHECKPOINT_DIR / "cost_projection_report.json").read_text(encoding="utf-8")
    )

    plot_recall_by_dimension(evaluation_report)
    latency = plot_latency_comparison(cost_projection_report)
    print(f"Wrote {ASSETS_DIR / 'recall_by_dimension.png'}")
    print(f"Wrote {ASSETS_DIR / 'latency_comparison.png'}")

    speedup_measured = latency["measured_brute_force_ms"] / latency["measured_funnel_ms"]
    speedup_extrapolated = (
        latency["extrapolated_brute_force_ms"] / latency["extrapolated_funnel_ms"]
    )
    print(
        "\nCopy these into README.md / docs/08_results.md prose verbatim — do not "
        "hand-type or re-derive latency numbers from a different script's report; "
        "that's what caused prose and chart to disagree last time:\n"
        f"  Measured ({latency['n_measured']:,} items): "
        f"{latency['measured_brute_force_ms']:.1f}ms -> "
        f"{latency['measured_funnel_ms']:.1f}ms (~{speedup_measured:.1f}x), "
        f"Stage 1 @ {latency['low_dim']}-dim\n"
        f"  Extrapolated ({latency['n_full']:,} items): "
        f"{latency['extrapolated_brute_force_ms']:.1f}ms -> "
        f"{latency['extrapolated_funnel_ms']:.1f}ms (~{speedup_extrapolated:.1f}x)"
    )


if __name__ == "__main__":
    main()
