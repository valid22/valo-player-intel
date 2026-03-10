from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from valo_player_intel.data.io import read_table
from valo_player_intel.data.manifest import SourceDefinition
from valo_player_intel.data.schemas import MATCH_COLUMNS, PLAYER_MATCH_COLUMNS


@dataclass(slots=True)
class LoadedSource:
    matches: pd.DataFrame
    player_matches: pd.DataFrame


def _apply_mapping(df: pd.DataFrame, mapping: dict[str, str], required_columns: list[str]) -> pd.DataFrame:
    renamed = df.rename(columns={source: target for target, source in mapping.items()})
    for column in required_columns:
        if column not in renamed.columns:
            renamed[column] = pd.NA
    return renamed[required_columns]


def load_csv_bundle(base_dir: Path, source: SourceDefinition) -> LoadedSource:
    layout = source.layout
    matches_df = read_table(base_dir / layout.matches_path)
    player_df = read_table(base_dir / layout.player_matches_path)

    matches = _apply_mapping(matches_df, layout.column_mapping, MATCH_COLUMNS)
    player_matches = _apply_mapping(player_df, layout.column_mapping, PLAYER_MATCH_COLUMNS)

    matches["cohort"] = source.cohort
    matches["source_name"] = source.source_name
    player_matches["cohort"] = source.cohort

    for key, value in layout.static_values.items():
        if key in matches.columns:
            matches[key] = value
        if key in player_matches.columns:
            player_matches[key] = value

    return LoadedSource(matches=matches, player_matches=player_matches)


def load_json_bundle(base_dir: Path, source: SourceDefinition) -> LoadedSource:
    layout = source.layout
    matches_payload = json.loads((base_dir / layout.matches_path).read_text())
    player_payload = json.loads((base_dir / layout.player_matches_path).read_text())

    matches = pd.DataFrame(matches_payload.get("matches", []))
    player_matches = pd.DataFrame(player_payload.get("player_matches", []))

    for column in MATCH_COLUMNS:
        if column not in matches.columns:
            matches[column] = pd.NA
    for column in PLAYER_MATCH_COLUMNS:
        if column not in player_matches.columns:
            player_matches[column] = pd.NA

    matches["cohort"] = source.cohort
    matches["source_name"] = source.source_name
    player_matches["cohort"] = source.cohort

    return LoadedSource(matches=matches[MATCH_COLUMNS], player_matches=player_matches[PLAYER_MATCH_COLUMNS + ["cohort"]])
