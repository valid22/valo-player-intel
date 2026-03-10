from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from valo_player_intel.data.schemas import Cohort


class CsvLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matches_path: str
    player_matches_path: str
    match_id_column: str = "match_id"
    player_id_column: str = "player_id"
    datetime_column: str | None = None
    column_mapping: dict[str, str] = Field(default_factory=dict)
    static_values: dict[str, Any] = Field(default_factory=dict)


class JsonLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matches_path: str
    player_matches_path: str


class SourceDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str
    cohort: Cohort
    format: Literal["csv_bundle", "json_bundle"] = "csv_bundle"
    description: str = ""
    layout: CsvLayout | JsonLayout


class SourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    sources: list[SourceDefinition]

    @classmethod
    def from_path(cls, path: str | Path) -> "SourceManifest":
        return cls.model_validate(json.loads(Path(path).read_text()))
