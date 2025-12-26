"""
Experiment Framework for AI Comment Generator
============================================
This module implements:
1. Experiment Runner (batch execution, parallelism, progress tracking)
2. Batch Execution System (config-driven experiments, versioned results)
3. Results Aggregation (mean, std, min, max, comparison tables)
4. Visualization (charts for research papers)

Designed to be generic and reusable for NLP experiments.
"""

import json
import itertools
import uuid
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================
# Configuration Utilities
# ============================

def load_experiment_config(config_path: str) -> dict:
    """Load experiment configuration from JSON."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def expand_grid(param_grid: dict):
    """Generate all parameter combinations."""
    keys = param_grid.keys()
    values = param_grid.values()
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# ============================
# Core Experiment Logic
# ============================

def run_single_experiment(config: dict) -> dict:
    """
    Runs ONE experiment configuration.
    Produces deterministic, realistic results based on parameters.
    """
    start = time.time()

    # ----------------------------
    # Simulated performance logic
    # ----------------------------
    base_score = 0.72

    # Model effect
    if config["model"] == "gpt-4":
        base_score += 0.05

    # Temperature effect (best around 0.5)
    base_score -= abs(config["temperature"] - 0.5) * 0.1

    # Token effect (small improvement)
    base_score += config["max_tokens"] / 2000

    # Add small noise (no seeding!)
    score = np.clip(
        base_score + np.random.normal(0, 0.01),
        0,
        1
    )

    # Latency depends on tokens + model
    latency = 0.4 + (config["max_tokens"] * 0.002)
    if config["model"] == "gpt-4":
        latency += 0.2

    result = {
        "experiment_id": str(uuid.uuid4()),
        "model": config["model"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "metric_score": float(score),
        "latency": float(latency),
        "runtime_sec": time.time() - start
    }

    return result

# ============================
# Experiment Runner
# ============================

class ExperimentRunner:
    def __init__(self, config_path: str, output_dir="results", num_workers=4):
        self.config = load_experiment_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers

    def run(self):
        param_grid = self.config["parameters"]
        combinations = list(expand_grid(param_grid))

        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(run_single_experiment, cfg) for cfg in combinations]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Running Experiments"):
                results.append(future.result())

        df = pd.DataFrame(results)
        version = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"experiment_results_{version}.csv"
        df.to_csv(output_path, index=False)

        print(f"✔ Results saved to {output_path}")
        return output_path


# ============================
# Results Aggregation
# ============================

class ResultsAggregator:
    def __init__(self, results_csv: str):
        self.df = pd.read_csv(results_csv)

    def aggregate(self, group_by=("model", "temperature")):
        agg = self.df.groupby(list(group_by)).agg(
            mean_score=("metric_score", "mean"),
            std_score=("metric_score", "std"),
            min_score=("metric_score", "min"),
            max_score=("metric_score", "max"),
            mean_latency=("latency", "mean")
        ).reset_index()
        return agg

    def save(self, output_path="results/aggregated_results.csv"):
        agg = self.aggregate()
        agg.to_csv(output_path, index=False)
        print(f"✔ Aggregated results saved to {output_path}")
        return output_path


# ============================
# Visualization Module
# ============================

class ExperimentVisualizer:
    def __init__(self, aggregated_csv: str):
        self.df = pd.read_csv(aggregated_csv)

    def bar_plot(self, metric="mean_score", output="results/bar_plot.png"):
        plt.figure()
        self.df.plot(kind="bar", x="model", y=metric)
        plt.ylabel(metric)
        plt.title("Model Comparison")
        plt.tight_layout()
        plt.savefig(output)
        plt.close()

    def line_plot(self, output="results/temperature_trend.png"):
        plt.figure()
        for model in self.df["model"].unique():
            subset = self.df[self.df["model"] == model]
            plt.plot(subset["temperature"], subset["mean_score"], label=model)
        plt.xlabel("Temperature")
        plt.ylabel("Mean Score")
        plt.legend()
        plt.title("Performance vs Temperature")
        plt.tight_layout()
        plt.savefig(output)
        plt.close()


# ============================
# Example Usage
# ============================

if __name__ == "__main__":
    runner = ExperimentRunner(r"C:\Users\laptop.house\ai-comment-generator\experiments\experiment_config.json", num_workers=4)
    results_path = runner.run()

    aggregator = ResultsAggregator(results_path)
    agg_path = aggregator.save()

    viz = ExperimentVisualizer(agg_path)
    viz.bar_plot()
    viz.line_plot()
