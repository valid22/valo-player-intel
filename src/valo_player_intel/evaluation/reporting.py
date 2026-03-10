from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

from valo_player_intel.data.io import write_json, write_table
from valo_player_intel.evaluation.metrics import expected_calibration_error, reliability_curve
from valo_player_intel.prediction.models import ModelResult
from valo_player_intel.segmentation.clustering import AgentBehaviorResult, CohortClusteringResult


def plot_reliability(curve: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = px.line(
        curve,
        x="mean_predicted",
        y="empirical_win_rate",
        markers=True,
        title=title,
        labels={"mean_predicted": "Predicted win probability", "empirical_win_rate": "Observed win rate"},
    )
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash"}, name="Perfect calibration")
    fig.write_html(path, include_plotlyjs="cdn")


def plot_pca(points: pd.DataFrame, path: Path, title: str, hover_data: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = px.scatter(points, x="pca_1", y="pca_2", color=hover_data[0], hover_data=hover_data[1:], title=title)
    fig.write_html(path, include_plotlyjs="cdn")


def write_outputs(
    clustering_results: list[CohortClusteringResult],
    role_clustering_results: list[CohortClusteringResult],
    agent_behavior_results: list[AgentBehaviorResult],
    cluster_outcomes: pd.DataFrame,
    cohort_comparison: pd.DataFrame,
    model_results: list[ModelResult],
    artifacts_dir: Path,
    figures_dir: Path,
    results_path: Path,
) -> None:
    _clear_generated_outputs(artifacts_dir, figures_dir)
    summary: dict = {"segmentation": {}, "prediction": {}, "role_segmentation": {}, "agent_behavior": {}, "findings": {}}

    for result in clustering_results:
        cluster_path = write_table(result.assignments, artifacts_dir / "segmentation" / f"{result.cohort}_clusters.parquet")
        profile_path = write_table(result.profile_table, artifacts_dir / "segmentation" / f"{result.cohort}_cluster_profiles.parquet")
        pca_path = artifacts_dir / "segmentation" / f"{result.cohort}_pca_points.parquet"
        write_table(result.pca_points, pca_path)
        plot_pca(result.pca_points, figures_dir / f"{result.cohort}_clusters.html", f"{result.cohort.title()} Player Archetypes", ["archetype_label", "player_name", "cluster_id"])
        summary["segmentation"][result.cohort] = {
            **result.metrics,
            "cluster_artifact": str(cluster_path),
            "profile_artifact": str(profile_path),
        }

    for result in role_clustering_results:
        group_slug = _slugify(result.group_name or "unknown")
        cluster_path = write_table(result.assignments, artifacts_dir / "segmentation" / f"{result.cohort}_{group_slug}_role_clusters.parquet")
        profile_path = write_table(result.profile_table, artifacts_dir / "segmentation" / f"{result.cohort}_{group_slug}_role_cluster_profiles.parquet")
        pca_path = artifacts_dir / "segmentation" / f"{result.cohort}_{group_slug}_role_pca_points.parquet"
        write_table(result.pca_points, pca_path)
        plot_pca(
            result.pca_points,
            figures_dir / f"{result.cohort}_{group_slug}_role_clusters.html",
            f"{result.cohort.title()} {result.group_name} Archetypes",
            ["archetype_label", "player_name", "cluster_id"],
        )
        summary["role_segmentation"].setdefault(result.cohort, {})[result.group_name or "unknown"] = {
            **result.metrics,
            "cluster_artifact": str(cluster_path),
            "profile_artifact": str(profile_path),
        }

    for result in agent_behavior_results:
        profile_path = write_table(result.agent_profiles, artifacts_dir / "segmentation" / f"{result.cohort}_agent_behavior_profiles.parquet")
        pca_path = artifacts_dir / "segmentation" / f"{result.cohort}_agent_behavior_pca_points.parquet"
        write_table(result.pca_points, pca_path)
        plot_pca(result.pca_points, figures_dir / f"{result.cohort}_agent_behavior.html", f"{result.cohort.title()} Agent Behavior Map", ["agent", "cluster_id"])
        summary["agent_behavior"][result.cohort] = {**result.metrics, "profile_artifact": str(profile_path)}

    if not cluster_outcomes.empty:
        cluster_outcomes_path = write_table(cluster_outcomes, artifacts_dir / "segmentation" / "cluster_outcomes.parquet")
    else:
        cluster_outcomes_path = artifacts_dir / "segmentation" / "cluster_outcomes.parquet"
    if not cohort_comparison.empty:
        cohort_comparison_path = write_table(cohort_comparison, artifacts_dir / "segmentation" / "cohort_comparison.parquet")
    else:
        cohort_comparison_path = artifacts_dir / "segmentation" / "cohort_comparison.parquet"

    calibration_rows = []
    prediction_rows = []
    model_metrics: dict[str, dict[str, dict[str, float | str]]] = {}
    for result in model_results:
        curve = reliability_curve(result.predictions["actual"], result.predictions["probability"])
        ece = expected_calibration_error(result.predictions["actual"], result.predictions["probability"])
        curve["cohort"] = result.cohort
        curve["model_name"] = result.model_name
        calibration_rows.append(curve)
        prediction_frame = result.predictions.copy()
        prediction_frame["cohort"] = result.cohort
        prediction_frame["model_name"] = result.model_name
        prediction_rows.append(prediction_frame)
        plot_reliability(curve, figures_dir / f"{result.cohort}_{result.model_name}_reliability.html", f"{result.cohort.title()} {result.model_name} Reliability")
        model_metrics.setdefault(result.cohort, {})[result.model_name] = {**result.metrics, "ece": ece}

    calibration_path = artifacts_dir / "prediction" / "calibration_curves.parquet"
    predictions_path = artifacts_dir / "prediction" / "model_predictions.parquet"
    if calibration_rows:
        calibration_path = write_table(pd.concat(calibration_rows, ignore_index=True), calibration_path)
    if prediction_rows:
        predictions_path = write_table(pd.concat(prediction_rows, ignore_index=True), predictions_path)

    metrics_path = artifacts_dir / "prediction" / "model_metrics.json"
    write_json(model_metrics, metrics_path)

    summary["prediction"] = {
        "model_metrics_artifact": str(metrics_path),
        "calibration_curves_artifact": str(calibration_path),
        "predictions_artifact": str(predictions_path),
        "cohorts": model_metrics,
        "best_models": _best_model_summary(model_metrics),
    }
    summary["findings"] = {
        "cluster_outcomes_artifact": str(cluster_outcomes_path),
        "cohort_comparison_artifact": str(cohort_comparison_path),
    }
    write_json(summary, results_path)


def _clear_generated_outputs(artifacts_dir: Path, figures_dir: Path) -> None:
    for root in (artifacts_dir / "segmentation", artifacts_dir / "prediction", figures_dir):
        if not root.exists():
            continue
        for path in root.glob("*"):
            if path.is_file() and path.name != ".gitkeep":
                path.unlink()


def _best_model_summary(model_metrics: dict[str, dict[str, dict[str, float | str]]]) -> dict[str, dict[str, float | str | None]]:
    summary: dict[str, dict[str, float | str | None]] = {}
    for cohort, metrics_by_model in model_metrics.items():
        if not metrics_by_model:
            continue
        ranked = sorted(metrics_by_model.items(), key=lambda item: item[1].get("brier_score", float("inf")))
        best_model_name, best_metrics = ranked[0]
        baseline_brier = metrics_by_model.get("static_baseline", {}).get("brier_score")
        best_brier = best_metrics.get("brier_score")
        brier_improvement = None
        if isinstance(baseline_brier, (int, float)) and isinstance(best_brier, (int, float)) and baseline_brier > 0:
            brier_improvement = 1 - (best_brier / baseline_brier)
        summary[cohort] = {
            "model_name": best_model_name,
            "brier_score": best_brier,
            "roc_auc": best_metrics.get("roc_auc"),
            "f1": best_metrics.get("f1"),
            "ece": best_metrics.get("ece"),
            "brier_improvement_vs_baseline": brier_improvement,
        }
    return summary


def _slugify(value: str) -> str:
    return value.lower().replace("/", "_").replace(" ", "_")
