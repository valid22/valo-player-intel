import json

from valo_player_intel.data.normalize import normalize_sources


def test_normalize_json_bundle(tmp_path):
    raw_vct = tmp_path / "raw" / "vct"
    raw_public = tmp_path / "raw" / "public"
    external = tmp_path / "external"
    raw_vct.mkdir(parents=True)
    raw_public.mkdir(parents=True)
    external.mkdir(parents=True)

    (raw_vct / "vlr_matches.json").write_text(
        json.dumps(
            {
                "matches": [
                    {
                        "match_id": "m1",
                        "match_datetime": "2025-01-01T00:00:00Z",
                        "cohort": "pro",
                        "source_name": "vlr_vct_public",
                        "map_name": "Ascent",
                        "team_a_id": "A",
                        "team_b_id": "B",
                        "winning_team_id": "A",
                        "best_of": 3,
                        "event_tier": "Tier1",
                    }
                ]
            }
        )
    )
    (raw_vct / "vlr_player_matches.json").write_text(
        json.dumps(
            {
                "player_matches": [
                    {
                        "match_id": "m1",
                        "player_id": "p1",
                        "player_name": "Alpha",
                        "team_id": "A",
                        "agent": "Jett",
                        "kills": 20,
                        "deaths": 10,
                        "assists": 5,
                        "won_match": 1,
                        "cohort": "pro",
                    }
                ]
            }
        )
    )
    (raw_public / "henrik_matches.json").write_text(json.dumps({"matches": []}))
    (raw_public / "henrik_player_matches.json").write_text(json.dumps({"player_matches": []}))
    (external / "source_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "sources": [
                    {
                        "source_name": "vlr_vct_public",
                        "cohort": "pro",
                        "format": "json_bundle",
                        "description": "",
                        "layout": {
                            "matches_path": "raw/vct/vlr_matches.json",
                            "player_matches_path": "raw/vct/vlr_player_matches.json",
                        },
                    },
                    {
                        "source_name": "henrik_public_competitive",
                        "cohort": "public",
                        "format": "json_bundle",
                        "description": "",
                        "layout": {
                            "matches_path": "raw/public/henrik_matches.json",
                            "player_matches_path": "raw/public/henrik_player_matches.json",
                        },
                    },
                ],
            }
        )
    )

    outputs = normalize_sources(external / "source_manifest.json", tmp_path / "interim")

    assert outputs["matches"].exists()
    assert outputs["player_matches"].exists()
