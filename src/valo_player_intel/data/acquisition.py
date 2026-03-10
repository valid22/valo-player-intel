from __future__ import annotations

from pathlib import Path

from valo_player_intel.data.io import write_json
from valo_player_intel.config.settings import PipelinePaths
from valo_player_intel.data.manifest import SourceManifest
from valo_player_intel.data.sources.henrik import HenrikConfig, fetch_public_matches
from valo_player_intel.data.sources.vlr import VLRConfig, fetch_vct_matches


def fetch_all_sources(paths: PipelinePaths, append: bool = True) -> dict[str, Path]:
    outputs = {}
    fetch_report: dict[str, dict[str, str | bool]] = {}

    try:
        outputs.update(fetch_vct_matches(VLRConfig(append=append), paths.data_raw_vct))
        fetch_report["vct"] = {"ok": True, "message": "Fetched VCT public data."}
    except Exception as exc:
        fetch_report["vct"] = {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    try:
        outputs.update(
            fetch_public_matches(
                HenrikConfig(
                    seed_players_path=paths.project_root / "data" / "external" / "public_seed_players.csv",
                    append=append,
                ),
                paths.data_raw_public,
            )
        )
        fetch_report["public"] = {"ok": True, "message": "Fetched public competitive match data."}
    except Exception as exc:
        fetch_report["public"] = {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    fetch_report["append_mode"] = {"ok": True, "message": str(append)}
    write_json(fetch_report, paths.data_raw / "fetch_report.json")
    return outputs


def build_default_manifest(paths: PipelinePaths) -> SourceManifest:
    return SourceManifest.model_validate_json(paths.source_manifest.read_text())
