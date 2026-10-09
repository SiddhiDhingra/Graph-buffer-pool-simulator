"""
Final research evaluation runner.
"""

from __future__ import annotations

from evaluation.experiments import (
    ExperimentConfig,
    run_research_suite,
)

from evaluation.plots import generate_evaluation_plots


if __name__ == "__main__":

    config = ExperimentConfig(
        requests=5000,

        capacities=(
            3,
            5,
            10,
        ),

        seeds=(
            7,
            19,
            42,
            73,
            101,
            137,
            211,
            307,
            401,
            503,
        ),
    )

    run_research_suite(
        config,
        output_dir="results",
    )

    generate_evaluation_plots()

    print()
    print("Research evaluation completed.")
    print("Raw results: results/raw_results.csv")
    print("Summary:     results/summary_results.csv")
    print("Plots:       results/plots/")