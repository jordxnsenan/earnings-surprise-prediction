import json
import numpy as np
import pandas as pd

from pathlib import Path

from earnings_surprise.clustering.dbscan import run_dbscan
from earnings_surprise.clustering.prepare import (
    prepare_clustering_data,
)
from earnings_surprise.data.samples import PREDICTION_WINDOWS_PATH
from earnings_surprise.features import FEATURE_MATRIX_PATH

DBSCAN_DIAGNOSTICS_PATH = Path("results/clustering/dbscan_distances.json")

DBSCAN_SWEEP_PATH = Path("results/clustering/dbscan_sweep.json")

EPS_PERCENTILES = ("50", "70", "85", "95")


def build_parameter_grid(diagnostics: dict) -> list[dict]:
    """builds DBSCAN configurations from measured neighbor distances"""
    if diagnostics.get("holdout_used") is not False:
        raise ValueError("Diagnostics must exclude the holdout")

    configurations = []
    seen_min_samples = set()

    for entry in diagnostics["diagnostics"]:
        min_samples = entry["min_samples"]

        if min_samples in seen_min_samples:
            raise ValueError("Duplicate min_samples configuration")

        seen_min_samples.add(min_samples)

        for percentile in EPS_PERCENTILES:
            eps = float(entry["distance_percentiles"][percentile])

            if not np.isfinite(eps) or eps <= 0:
                raise ValueError("Invalid neighborhood distance")

            configurations.append(
                {
                    "min_samples": min_samples,
                    "eps_percentile": int(percentile),
                    "eps": eps,
                }
            )

    if not configurations:
        raise ValueError("No DBSCAN configurations available")

    return configurations


def evaluate_dbscan_grid(
    data: np.ndarray,
    configurations: list[dict],
) -> list[dict]:
    """evaluates DBSCAN configurations on the same observations"""
    results = []

    for configuration in configurations:
        _, summary = run_dbscan(
            data,
            eps=configuration["eps"],
            min_samples=configuration["min_samples"],
        )

        summary["eps_percentile"] = configuration["eps_percentile"]

        results.append(summary)

    return results


def main() -> None:
    """evaluates DBSCAN using the saved distance diagnostics"""
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    samples = pd.read_parquet(PREDICTION_WINDOWS_PATH)

    transformed, selected_positions, _ = prepare_clustering_data(
        features,
        samples,
    )

    with DBSCAN_DIAGNOSTICS_PATH.open(encoding="utf-8") as file:
        diagnostics = json.load(file)

    if (
        diagnostics["sample_count"] != len(transformed)
        or diagnostics["feature_count"] != transformed.shape[1]
    ):
        raise ValueError("Distance diagnostics do not match clustering data")

    configurations = build_parameter_grid(diagnostics)
    evaluations = evaluate_dbscan_grid(
        transformed,
        configurations,
    )

    report = {
        "experiment": "dbscan_parameter_comparison",
        "sample_count": len(selected_positions),
        "feature_count": transformed.shape[1],
        "holdout_used": False,
        "evaluation_population": "clustered_samples_only",
        "evaluations": evaluations,
    }

    DBSCAN_SWEEP_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DBSCAN_SWEEP_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("DBSCAN Parameter Comparison")
    print("---------------------------")

    for result in evaluations:
        print(f"\nmin_samples={result['min_samples']}, " f"eps={result['eps']:.4f}")

        print(f"  Clusters: {result['cluster_count']}")
        print(f"  Noise: {result['noise_fraction']:.1%}")
        print(f"  Clustered samples: {result['clustered_samples']:,}")

        silhouette = result["silhouette_score"]

        if silhouette is None:
            print("  Silhouette: undefined")
        else:
            print(f"  Silhouette: {silhouette:.4f}")

        print(f"  Cluster sizes: {result['cluster_sizes']}")

    print(f"\nResults saved to: {DBSCAN_SWEEP_PATH}")


if __name__ == "__main__":
    main()