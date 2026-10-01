from __future__ import annotations
import matplotlib.pyplot as plt
from evaluation.experiments import run_comparison_experiment

def generate_evaluation_plots():
    print("📈 Running experiments and generating visual comparison charts...")
    results = run_comparison_experiment()
    
    policies = ["fifo", "lru", "proposed"]
    x_labels = [p.upper() for p in policies]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#ff9999", "#66b3ff", "#99ff99"]

    # 1. Hit Ratio Bar Chart
    hr_vals = [results[p]["Hit Ratio"] for p in policies]
    axes[0].bar(x_labels, hr_vals, color=colors)
    axes[0].set_title("Hit Ratio Comparison (%)")
    axes[0].set_ylabel("Hit Ratio (%)")
    axes[0].set_ylim(0, 100)

    # 2. Page Faults Bar Chart
    pf_vals = [results[p]["Page Faults"] for p in policies]
    axes[1].bar(x_labels, pf_vals, color=colors)
    axes[1].set_title("Total Page Faults (Lower is Better)")
    axes[1].set_ylabel("Fault Count")

    # 3. Simulated Latency Bar Chart
    lat_vals = [results[p]["Simulated Latency"] for p in policies]
    axes[2].bar(x_labels, lat_vals, color=colors)
    axes[2].set_title("Simulated Cost / Latency")
    axes[2].set_ylabel("Cost Units")

    plt.tight_layout()
    plt.savefig("evaluation/performance_comparison.png")
    print("✅ Success! Plot saved to evaluation/performance_comparison.png")
    plt.show()

if __name__ == "__main__":
    generate_evaluation_plots()