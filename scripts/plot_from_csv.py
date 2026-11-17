"""Build figures from measured experiment_results.csv; never create mock data."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "experiment_results.csv"


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"{SOURCE} is missing. Run scripts/run_factorial_experiment.py first. "
            "This plotting script does not generate synthetic result values."
        )

    frame = pd.read_csv(SOURCE)
    required = {
        "system", "k_docs", "position", "h1_qualifier_recall_pct",
        "h4_contradiction_pct", "h3_nli_faithfulness_pct",
        "headline_recall_pct", "latency_sec",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    figure_specs = [
        ("h1_qualifier_recall_pct", "H1 qualifier retention (%)",
         "fig1_positional_degradation"),
        ("h4_contradiction_pct", "H4 contradiction detection (%)",
         "fig2_contradiction_detection"),
        ("h3_nli_faithfulness_pct", "H3 NLI citation faithfulness (%)",
         "fig3_citation_faithfulness"),
        ("latency_sec", "Wall-clock latency (seconds)",
         "fig4_latency"),
    ]
    position_order = ["start", "middle", "end"]

    for metric, y_label, stem in figure_specs:
        grouped = (
            frame.groupby(["system", "k_docs", "position"], as_index=False)[metric]
            .mean()
        )
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        for (system, k_docs), subset in grouped.groupby(["system", "k_docs"]):
            subset = subset.set_index("position").reindex(position_order)
            ax.plot(
                position_order, subset[metric], marker="o",
                label=f"{system}, k={k_docs}",
            )
        ax.set_xlabel("Target-pair position")
        ax.set_ylabel(y_label)
        ax.set_title(y_label + " by system and corpus size")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(RESULTS / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(RESULTS / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {stem}.png and {stem}.pdf")

    print("All figures were derived from measured rows in experiment_results.csv.")


if __name__ == "__main__":
    main()
