import json

from pathlib import Path

RESULT_PATHS = {
    "kmeans": Path("results/clustering/kmeans_sweep.json"),
    "agglomerative": Path("results/clustering/agglomerative_sweep.json"),
    "dbscan": Path("results/clustering/dbscan_sweep.json"),
    "partitions": Path("results/clustering/partition_comparison.json"),
}

COMPARISON_OUTPUT_PATH = Path("results/clustering/algorithm_comparison.json")

SELECTED_CLUSTER_COUNT = 2


def find_cluster_result(report: dict, cluster_count: int) -> dict:
    """finds one clustering result with the requested cluster count"""
    matches = [
        result
        for result in report["evaluations"]
        if result["cluster_count"] == cluster_count
    ]

    if len(matches) != 1:
        raise ValueError("Missing or duplicate cluster configuration")

    return matches[0]


def build_algorithm_comparison(reports: dict) -> dict:
    """compares saved clustering results on the same sample"""
    if set(reports) != set(RESULT_PATHS):
        raise ValueError("Missing clustering reports")

    partition_report = reports["partitions"]

    sample_count = partition_report["samples"]
    feature_count = partition_report["feature_count"]

    for name, report in reports.items():
        if name == "partitions":
            continue

        if (
            report["sample_count"] != sample_count
            or report["feature_count"] != feature_count
        ):
            raise ValueError("Clustering sample dimensions do not match")

        holdout_used = (
            report["sampling"]["holdout_used"]
            if name in ("kmeans", "agglomerative")
            else report["holdout_used"]
        )

        if holdout_used is not False:
            raise ValueError("Clustering reports must exclude the holdout")

    if partition_report["holdout_used"] is not False:
        raise ValueError("Partition comparison must exclude the holdout")

    kmeans = find_cluster_result(
        reports["kmeans"],
        SELECTED_CLUSTER_COUNT,
    )

    agglomerative = find_cluster_result(
        reports["agglomerative"],
        SELECTED_CLUSTER_COUNT,
    )

    dbscan_evaluations = reports["dbscan"]["evaluations"]

    if not dbscan_evaluations:
        raise ValueError("No DBSCAN results available")

    for result in (kmeans, agglomerative):
        if sum(result["cluster_sizes"].values()) != sample_count:
            raise ValueError("Cluster sizes do not match sample count")

    contingency = partition_report["contingency_matrix"]

    if sum(sum(row) for row in contingency) != sample_count:
        raise ValueError("Partition comparison counts do not match")

    return {
        "experiment": "clustering_algorithm_comparison",
        "sample_count": sample_count,
        "feature_count": feature_count,
        "holdout_used": False,
        "kmeans": {
            "cluster_count": SELECTED_CLUSTER_COUNT,
            "silhouette_score": kmeans["silhouette_score"],
            "calinski_harabasz_score": (kmeans["calinski_harabasz_score"]),
            "davies_bouldin_score": kmeans["davies_bouldin_score"],
            "cluster_sizes": kmeans["cluster_sizes"],
        },
        "agglomerative": {
            "cluster_count": SELECTED_CLUSTER_COUNT,
            "silhouette_score": agglomerative["silhouette_score"],
            "calinski_harabasz_score": (agglomerative["calinski_harabasz_score"]),
            "davies_bouldin_score": (agglomerative["davies_bouldin_score"]),
            "cluster_sizes": agglomerative["cluster_sizes"],
        },
        "dbscan": {
            "configurations_tested": len(dbscan_evaluations),
            "cluster_counts": sorted(
                {result["cluster_count"] for result in dbscan_evaluations}
            ),
            "minimum_noise_fraction": min(
                result["noise_fraction"] for result in dbscan_evaluations
            ),
            "maximum_noise_fraction": max(
                result["noise_fraction"] for result in dbscan_evaluations
            ),
            "valid_silhouette_results": sum(
                result["silhouette_score"] is not None for result in dbscan_evaluations
            ),
        },
        "partition_agreement": {
            "adjusted_rand_index": (partition_report["adjusted_rand_index"]),
            "contingency_matrix": contingency,
        },
    }


def main() -> None:
    """consolidates the saved clustering experiments"""
    reports = {}

    for name, path in RESULT_PATHS.items():
        with path.open(encoding="utf-8") as file:
            reports[name] = json.load(file)

    comparison = build_algorithm_comparison(reports)

    COMPARISON_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    COMPARISON_OUTPUT_PATH.write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Clustering Algorithm Comparison")
    print("-------------------------------")

    for name in ("kmeans", "agglomerative"):
        result = comparison[name]

        print(f"\n{name.capitalize()}")
        print(f"  Silhouette: {result['silhouette_score']:.4f}")
        print("  Calinski-Harabasz: " f"{result['calinski_harabasz_score']:.2f}")
        print("  Davies-Bouldin: " f"{result['davies_bouldin_score']:.4f}")

    print("\nDBSCAN")
    print("  Cluster counts: " f"{comparison['dbscan']['cluster_counts']}")
    print(
        "  Noise range: "
        f"{comparison['dbscan']['minimum_noise_fraction']:.2%} "
        "to "
        f"{comparison['dbscan']['maximum_noise_fraction']:.2%}"
    )

    print(
        "\nAdjusted Rand Index: "
        f"{comparison['partition_agreement']['adjusted_rand_index']:.4f}"
    )

    print(f"\nSaved to: {COMPARISON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
