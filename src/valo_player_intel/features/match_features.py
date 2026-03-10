from __future__ import annotations

import numpy as np
import pandas as pd


def build_match_level_dataset(
    matches: pd.DataFrame,
    player_matches: pd.DataFrame,
    global_assignments: pd.DataFrame | None = None,
    role_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    player_df = player_matches.copy()
    numeric_columns = [
        "acs",
        "adr",
        "headshot_rate",
        "assists",
        "plants",
        "defuses",
        "first_kills",
        "first_deaths",
    ]
    for column in numeric_columns:
        if column not in player_df.columns:
            player_df[column] = np.nan
        player_df[column] = pd.to_numeric(player_df[column], errors="coerce")
    player_df["player_rating_proxy"] = (
        player_df["acs"].fillna(0) * 0.45
        + player_df["adr"].fillna(0) * 0.35
        + player_df["headshot_rate"].fillna(0) * 100 * 0.20
    )
    player_df["support_score_match"] = player_df["assists"].fillna(0) + player_df["plants"].fillna(0) + player_df["defuses"].fillna(0)
    player_df["objective_score_match"] = player_df["plants"].fillna(0) + player_df["defuses"].fillna(0)
    player_df["entry_power_match"] = player_df["first_kills"].fillna(0) - player_df["first_deaths"].fillna(0)

    if global_assignments is not None and not global_assignments.empty:
        global_lookup = global_assignments[["cohort", "player_id", "archetype_label"]].rename(columns={"archetype_label": "global_archetype_label"})
        player_df = player_df.merge(global_lookup, on=["cohort", "player_id"], how="left")

    if role_assignments is not None and not role_assignments.empty:
        role_lookup = role_assignments[["cohort", "player_id", "group_name", "archetype_label"]].rename(
            columns={"group_name": "role_cluster_group", "archetype_label": "role_archetype_label"}
        )
        player_df = player_df.merge(role_lookup, on=["cohort", "player_id"], how="left")

    team_features = (
        player_df.groupby(["match_id", "team_id"], dropna=False)
        .agg(
            team_rating_mean=("player_rating_proxy", "mean"),
            team_rating_std=("player_rating_proxy", "std"),
            team_entry_power=("entry_power_match", "mean"),
            team_support_power=("support_score_match", "mean"),
            team_objective_power=("objective_score_match", "mean"),
            team_consistency_mean=("acs", "std"),
            team_agent_diversity=("agent", lambda s: s.nunique()),
            team_role_balance_score=("agent_role", lambda s: s.nunique()),
        )
        .reset_index()
    )

    if "global_archetype_label" in player_df.columns:
        archetype_counts = _pivot_team_counts(player_df, "global_archetype_label", "team_global_arch")
        team_features = team_features.merge(archetype_counts, on=["match_id", "team_id"], how="left")
        global_cols = [col for col in archetype_counts.columns if col not in {"match_id", "team_id"}]
        if global_cols:
            team_features["team_global_archetype_diversity"] = team_features[global_cols].gt(0).sum(axis=1)
            team_features["team_global_archetype_balance"] = team_features[global_cols].std(axis=1)

    if "role_archetype_label" in player_df.columns:
        role_counts = _pivot_team_counts(player_df, "role_archetype_label", "team_role_arch")
        team_features = team_features.merge(role_counts, on=["match_id", "team_id"], how="left")
        role_cols = [col for col in role_counts.columns if col not in {"match_id", "team_id"}]
        if role_cols:
            team_features["team_role_archetype_diversity"] = team_features[role_cols].gt(0).sum(axis=1)
            team_features["team_role_archetype_balance"] = team_features[role_cols].std(axis=1)

    merged = matches.merge(
        team_features.add_prefix("a_"),
        left_on=["match_id", "team_a_id"],
        right_on=["a_match_id", "a_team_id"],
        how="left",
    ).merge(
        team_features.add_prefix("b_"),
        left_on=["match_id", "team_b_id"],
        right_on=["b_match_id", "b_team_id"],
        how="left",
    )

    a_columns = [col for col in merged.columns if col.startswith("a_team_global_arch__") or col.startswith("a_team_role_arch__")]
    output_rows = []
    for _, row in merged.iterrows():
        for team_label, opp_label in (("a", "b"), ("b", "a")):
            team_id = row[f"team_{team_label}_id"]
            base_row = {
                "match_id": row["match_id"],
                "cohort": row["cohort"],
                "source_name": row["source_name"],
                "map_name": row["map_name"],
                "event_tier": row.get("event_tier"),
                "team_id": team_id,
                "opponent_team_id": row[f"team_{opp_label}_id"],
                "won_match": int(team_id == row["winning_team_id"]),
                "team_rating_mean": row.get(f"{team_label}_team_rating_mean"),
                "team_rating_std": row.get(f"{team_label}_team_rating_std"),
                "team_entry_power": row.get(f"{team_label}_team_entry_power"),
                "team_support_power": row.get(f"{team_label}_team_support_power"),
                "team_objective_power": row.get(f"{team_label}_team_objective_power"),
                "team_consistency_mean": row.get(f"{team_label}_team_consistency_mean"),
                "team_agent_diversity": row.get(f"{team_label}_team_agent_diversity"),
                "team_role_balance_score": row.get(f"{team_label}_team_role_balance_score"),
                "team_global_archetype_diversity": row.get(f"{team_label}_team_global_archetype_diversity"),
                "team_global_archetype_balance": row.get(f"{team_label}_team_global_archetype_balance"),
                "team_role_archetype_diversity": row.get(f"{team_label}_team_role_archetype_diversity"),
                "team_role_archetype_balance": row.get(f"{team_label}_team_role_archetype_balance"),
                "opponent_gap_rating_mean": row.get(f"{team_label}_team_rating_mean") - row.get(f"{opp_label}_team_rating_mean"),
                "opponent_gap_entry_power": row.get(f"{team_label}_team_entry_power") - row.get(f"{opp_label}_team_entry_power"),
                "opponent_gap_support_power": row.get(f"{team_label}_team_support_power") - row.get(f"{opp_label}_team_support_power"),
                "opponent_gap_objective_power": row.get(f"{team_label}_team_objective_power") - row.get(f"{opp_label}_team_objective_power"),
                "opponent_gap_global_archetype_diversity": _safe_diff(row.get(f"{team_label}_team_global_archetype_diversity"), row.get(f"{opp_label}_team_global_archetype_diversity")),
                "opponent_gap_role_archetype_diversity": _safe_diff(row.get(f"{team_label}_team_role_archetype_diversity"), row.get(f"{opp_label}_team_role_archetype_diversity")),
            }
            for a_column in a_columns:
                feature_suffix = a_column[2:]
                base_row[feature_suffix.replace("team_", "")] = row.get(f"{team_label}_{feature_suffix}")
                base_row[f"opponent_gap_{feature_suffix.replace('team_', '')}"] = _safe_diff(
                    row.get(f"{team_label}_{feature_suffix}"),
                    row.get(f"{opp_label}_{feature_suffix}"),
                )
            output_rows.append(base_row)

    match_level = pd.DataFrame(output_rows)
    match_level["team_form_recent"] = (
        match_level.sort_values("match_id")
        .groupby(["cohort", "team_id"])["team_rating_mean"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    match_level["team_map_strength_proxy"] = match_level.groupby(["cohort", "team_id", "map_name"])["won_match"].transform("mean")
    return match_level.replace([np.inf, -np.inf], np.nan)


def _pivot_team_counts(player_df: pd.DataFrame, label_column: str, prefix: str) -> pd.DataFrame:
    subset = player_df.dropna(subset=[label_column]).copy()
    if subset.empty:
        return pd.DataFrame(columns=["match_id", "team_id"])
    counts = subset.groupby(["match_id", "team_id", label_column]).size().reset_index(name="count")
    pivot = counts.pivot_table(index=["match_id", "team_id"], columns=label_column, values="count", fill_value=0)
    pivot.columns = [f"{prefix}__{_sanitize_column(str(column))}" for column in pivot.columns]
    return pivot.reset_index()


def _sanitize_column(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(",", "")
        .replace("%", "pct")
        .replace("__", "_")
    )


def _safe_diff(a: float | None, b: float | None) -> float | None:
    if pd.isna(a) and pd.isna(b):
        return np.nan
    return (0.0 if pd.isna(a) else float(a)) - (0.0 if pd.isna(b) else float(b))
