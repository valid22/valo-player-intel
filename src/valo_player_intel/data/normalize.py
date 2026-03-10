from __future__ import annotations

from pathlib import Path

import pandas as pd

from valo_player_intel.data.io import write_json, write_table
from valo_player_intel.data.loaders import load_csv_bundle, load_json_bundle
from valo_player_intel.data.manifest import SourceManifest
from valo_player_intel.data.schemas import MATCH_COLUMNS, PLAYER_MATCH_COLUMNS
from valo_player_intel.features.player_features import normalize_agent_name


def _parse_mixed_datetime(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    parsed_numeric = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
    text_mask = numeric.isna()
    parsed_text = pd.to_datetime(series.where(text_mask), errors="coerce", utc=True, format="mixed")
    return parsed_numeric.fillna(parsed_text)


def normalize_sources(manifest_path: Path, output_dir: Path) -> dict[str, Path]:
    manifest = SourceManifest.from_path(manifest_path)
    base_dir = manifest_path.parent.parent
    all_matches: list[pd.DataFrame] = []
    all_player_matches: list[pd.DataFrame] = []

    for source in manifest.sources:
        layout = source.layout
        paths_to_check = []
        if hasattr(layout, "matches_path"):
            paths_to_check.append(base_dir / layout.matches_path)
        if hasattr(layout, "player_matches_path"):
            paths_to_check.append(base_dir / layout.player_matches_path)
        if not all(path.exists() for path in paths_to_check):
            continue

        if source.format == "csv_bundle":
            loaded = load_csv_bundle(base_dir, source)
        else:
            loaded = load_json_bundle(base_dir, source)
        all_matches.append(loaded.matches[MATCH_COLUMNS])
        all_player_matches.append(loaded.player_matches[PLAYER_MATCH_COLUMNS + ["cohort"]])

    if not all_matches or not all_player_matches:
        raise FileNotFoundError("No configured sources were available to normalize.")

    all_matches = [frame.dropna(axis=1, how="all") for frame in all_matches if not frame.empty]
    all_player_matches = [frame.dropna(axis=1, how="all") for frame in all_player_matches if not frame.empty]
    matches = pd.concat(all_matches, ignore_index=True).reindex(columns=MATCH_COLUMNS).drop_duplicates(subset=["match_id"])
    player_matches = pd.concat(all_player_matches, ignore_index=True).reindex(
        columns=PLAYER_MATCH_COLUMNS + ["cohort"]
    ).drop_duplicates(
        subset=["match_id", "player_id"]
    )
    if "agent" in player_matches.columns:
        player_matches["agent"] = player_matches["agent"].map(normalize_agent_name)
    matches["match_datetime"] = _parse_mixed_datetime(matches["match_datetime"])

    coverage = {
        "matches": matches.notna().mean().round(4).to_dict(),
        "player_matches": player_matches.notna().mean().round(4).to_dict(),
    }

    matches_path = write_table(matches, output_dir / "matches.parquet")
    player_path = write_table(player_matches, output_dir / "player_matches.parquet")
    coverage_path = output_dir / "coverage_report.json"
    write_json(coverage, coverage_path)

    return {
        "matches": matches_path,
        "player_matches": player_path,
        "coverage_report": coverage_path,
    }
