import pandas as pd

from valo_player_intel.features.match_features import build_match_level_dataset


def test_build_match_level_dataset_returns_two_rows_per_match():
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "cohort": "public",
                "source_name": "fixture",
                "map_name": "Ascent",
                "team_a_id": "A",
                "team_b_id": "B",
                "winning_team_id": "A",
            }
        ]
    )
    players = pd.DataFrame(
        [
            {"match_id": "m1", "team_id": "A", "agent_role": "Duelist", "agent": "Jett", "acs": 200, "adr": 140, "headshot_rate": 0.25, "assists": 3, "plants": 1, "defuses": 0, "first_kills": 2, "first_deaths": 1},
            {"match_id": "m1", "team_id": "A", "agent_role": "Controller", "agent": "Omen", "acs": 180, "adr": 120, "headshot_rate": 0.18, "assists": 8, "plants": 0, "defuses": 0, "first_kills": 0, "first_deaths": 1},
            {"match_id": "m1", "team_id": "B", "agent_role": "Duelist", "agent": "Raze", "acs": 190, "adr": 135, "headshot_rate": 0.20, "assists": 4, "plants": 0, "defuses": 0, "first_kills": 1, "first_deaths": 2},
            {"match_id": "m1", "team_id": "B", "agent_role": "Sentinel", "agent": "Cypher", "acs": 170, "adr": 110, "headshot_rate": 0.16, "assists": 6, "plants": 0, "defuses": 1, "first_kills": 0, "first_deaths": 1},
        ]
    )

    result = build_match_level_dataset(matches, players)
    assert len(result) == 2
    assert set(result["won_match"]) == {0, 1}
