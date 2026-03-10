from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PipelinePaths:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])

    @property
    def data_raw(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_raw_vct(self) -> Path:
        return self.data_raw / "vct"

    @property
    def data_raw_public(self) -> Path:
        return self.data_raw / "public"

    @property
    def data_interim(self) -> Path:
        return self.project_root / "data" / "interim"

    @property
    def data_processed(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def data_external(self) -> Path:
        return self.project_root / "data" / "external"

    @property
    def artifacts(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def figures(self) -> Path:
        return self.project_root / "reports" / "figures"

    @property
    def source_manifest(self) -> Path:
        return self.project_root / "data" / "external" / "source_manifest.json"

    @property
    def results_json(self) -> Path:
        return self.project_root / "results.json"


@dataclass(slots=True)
class ModelingConfig:
    random_state: int = 7
    min_feature_coverage: float = 0.70
    winsor_lower_quantile: float = 0.01
    winsor_upper_quantile: float = 0.99
    train_fraction: float = 0.80
    cluster_candidates: tuple[int, ...] = (4, 5, 6, 7, 8)
    default_cluster_count: int = 5
    calibration_bins: int = 10
