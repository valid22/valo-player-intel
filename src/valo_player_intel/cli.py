from __future__ import annotations

import argparse

import pandas as pd

from valo_player_intel.app.streamlit_app import run_app
from valo_player_intel.config.settings import ModelingConfig, PipelinePaths
from valo_player_intel.data.acquisition import fetch_all_sources
from valo_player_intel.data.io import read_table, write_table
from valo_player_intel.data.normalize import normalize_sources
from valo_player_intel.evaluation.reporting import write_outputs
from valo_player_intel.features.match_features import build_match_level_dataset
from valo_player_intel.features.player_features import build_player_features
from valo_player_intel.prediction.models import train_sklearn_models, train_torch_benchmark
from valo_player_intel.segmentation.clustering import (
    analyze_agent_behavior,
    compare_cohorts,
    run_clustering,
    run_role_conditioned_clustering,
    summarize_cluster_outcomes,
)


def run_pipeline(manifest: str) -> None:
    paths = PipelinePaths()
    config = ModelingConfig()
    normalized = normalize_sources(paths.project_root / manifest, paths.data_interim)

    matches = read_table(normalized["matches"])
    player_matches = read_table(normalized["player_matches"])
    player_features = build_player_features(player_matches, min_feature_coverage=config.min_feature_coverage)

    clustering_results = run_clustering(
        player_features=player_features,
        cluster_candidates=config.cluster_candidates,
        default_cluster_count=config.default_cluster_count,
        random_state=config.random_state,
    )
    role_clustering_results = run_role_conditioned_clustering(
        player_features=player_features,
        cluster_candidates=config.cluster_candidates,
        default_cluster_count=config.default_cluster_count,
        random_state=config.random_state,
    )
    combined_global_assignments = (
        pd.concat([result.assignments for result in clustering_results], ignore_index=True) if clustering_results else None
    )
    combined_role_assignments = (
        pd.concat([result.assignments for result in role_clustering_results], ignore_index=True) if role_clustering_results else None
    )
    match_level = build_match_level_dataset(matches, player_matches, combined_global_assignments, combined_role_assignments)
    agent_behavior_results = analyze_agent_behavior(player_features, random_state=config.random_state)
    cluster_outcomes = summarize_cluster_outcomes(player_features, clustering_results)
    cohort_comparison = compare_cohorts(clustering_results)

    write_table(player_features, paths.data_processed / "player_features.parquet")
    write_table(match_level, paths.data_processed / "match_level.parquet")
    model_results = train_sklearn_models(match_level) + train_torch_benchmark(match_level)

    write_outputs(
        clustering_results,
        role_clustering_results,
        agent_behavior_results,
        cluster_outcomes,
        cohort_comparison,
        model_results,
        paths.artifacts,
        paths.figures,
        paths.results_json,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VALORANT Player Intelligence pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch raw real-world VCT and public-match data")
    fetch_parser.add_argument("--no-append", action="store_true", help="Overwrite existing raw source files instead of appending.")

    run_parser = subparsers.add_parser("run", help="Run the full offline pipeline")
    run_parser.add_argument("--manifest", default="data/external/source_manifest.json")

    app_parser = subparsers.add_parser("app", help="Run the Streamlit app locally")
    app_parser.add_argument("--project-root", default=".")

    args = parser.parse_args()
    if args.command == "fetch":
        fetch_all_sources(PipelinePaths(), append=not args.no_append)
    elif args.command == "run":
        run_pipeline(args.manifest)
    else:
        run_app(PipelinePaths(project_root=args.project_root).project_root)


if __name__ == "__main__":
    main()
