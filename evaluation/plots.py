"""Publication-oriented plots from evaluation/summary_results.csv."""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def load_summary(path: str = "results/summary_results.csv"):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def generate_evaluation_plots(summary_path: str = "results/summary_results.csv",
                              output_dir: str = "results/plots"):
    rows = load_summary(summary_path)
    os.makedirs(output_dir, exist_ok=True)

    # Keep plots focused: one workload at a time makes paper figures readable.
    for workload in sorted({r["Workload"] for r in rows}):
        subset = [r for r in rows if r["Workload"] == workload]
        capacities = sorted({int(r["Capacity"]) for r in subset})
        policies = ["LRU", "Graph_Only", "LSH_Only", "Graph_LSH", "Proposed"]

        plt.figure(figsize=(8, 5))
        for policy in policies:
            points = [r for r in subset if r["Policy"] == policy]
            points.sort(key=lambda r: int(r["Capacity"]))
            if points:
                plt.plot(
                    [int(r["Capacity"]) for r in points],
                    [float(r["Hit Ratio Mean (%)"]) for r in points],
                    marker="o", label=policy,
                )
        plt.xlabel("Buffer capacity (pages)")
        plt.ylabel("Hit ratio (%)")
        plt.title(f"{workload}: Buffer Hit Ratio")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        safe = workload.lower().replace(" ", "_").replace("+", "plus")
        plt.savefig(os.path.join(output_dir, f"{safe}_hit_ratio.png"), dpi=300)
        plt.close()

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    generate_evaluation_plots()
