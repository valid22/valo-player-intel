import pandas as pd

from valo_player_intel.features.player_features import build_player_features


def test_build_player_features_generates_core_metrics():
    df = pd.DataFrame(
        [
            {"cohort": "pro", "match_id": "m1", "player_id": "p1", "player_name": "Alpha", "agent": "Jett", "kills": 20, "deaths": 10, "assists": 5, "headshot_rate": 0.3, "damage": 150, "adr": 140, "acs": 250, "kast": 0.72, "first_kills": 4, "first_deaths": 2, "plants": 0, "defuses": 0, "econ_spend": 4000, "utility_used": 2, "utility_damage": 10, "won_match": 1},
            {"cohort": "pro", "match_id": "m2", "player_id": "p1", "player_name": "Alpha", "agent": "Jett", "kills": 18, "deaths": 9, "assists": 7, "headshot_rate": 0.28, "damage": 160, "adr": 145, "acs": 245, "kast": 0.7, "first_kills": 3, "first_deaths": 1, "plants": 0, "defuses": 0, "econ_spend": 3900, "utility_used": 2, "utility_damage": 8, "won_match": 1},
        ]
    )

    features = build_player_features(df, min_feature_coverage=0.0)
    row = features.iloc[0]

    assert row["matches_played"] == 2
    assert row["win_rate"] == 1.0
    assert row["kda_ratio"] > 2.0
    assert "agent_role_entropy" in features.columns
