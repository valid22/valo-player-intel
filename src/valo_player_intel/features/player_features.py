from __future__ import annotations

import math

import numpy as np
import pandas as pd


ROLE_MAP = {
    "Jett": "Duelist",
    "Raze": "Duelist",
    "Reyna": "Duelist",
    "Yoru": "Duelist",
    "Phoenix": "Duelist",
    "Neon": "Duelist",
    "Iso": "Duelist",
    "Sova": "Initiator",
    "Skye": "Initiator",
    "KAY/O": "Initiator",
    "Kayo": "Initiator",
    "Fade": "Initiator",
    "Gekko": "Initiator",
    "Breach": "Initiator",
    "Killjoy": "Sentinel",
    "Cypher": "Sentinel",
    "Sage": "Sentinel",
    "Chamber": "Sentinel",
    "Deadlock": "Sentinel",
    "Vyse": "Sentinel",
    "Astra": "Controller",
    "Brimstone": "Controller",
    "Omen": "Controller",
    "Harbor": "Controller",
    "Viper": "Controller",
    "Clove": "Controller",
    "Tejo": "Initiator",
    "Waylay": "Duelist",
}


AGENT_NAME_MAP = {
    "KAYO": "KAY/O",
    "Kayo": "KAY/O",
    "Kay/O": "KAY/O",
    "kay/o": "KAY/O",
    "kay0": "KAY/O",
}

VALID_AGENTS = set(ROLE_MAP)


def normalize_agent_name(value: object) -> object:
    if value is None or value is pd.NA:
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    text = AGENT_NAME_MAP.get(text, text)
    return text if text in VALID_AGENTS else pd.NA


def _safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    result = np.where((denom.fillna(0) != 0), num.fillna(0) / denom.fillna(0), np.nan)
    return pd.Series(result, index=num.index, dtype="float64")


def _entropy(values: pd.Series) -> float:
    counts = values.value_counts(normalize=True)
    if counts.empty:
        return np.nan
    return float(-(counts * np.log2(counts)).sum())


def _mode_share(values: pd.Series) -> float:
    counts = values.value_counts(normalize=True)
    if counts.empty:
        return np.nan
    return float(counts.iloc[0])


def _mode_value(values: pd.Series) -> str | None:
    mode = values.dropna().astype(str).str.strip()
    mode = mode[mode != ""]
    if mode.empty:
        return None
    return str(mode.value_counts().idxmax())


def build_player_features(player_matches: pd.DataFrame, min_feature_coverage: float = 0.70) -> pd.DataFrame:
    df = player_matches.copy()
    numeric_columns = [
        "kills",
        "deaths",
        "assists",
        "headshot_rate",
        "damage",
        "adr",
        "acs",
        "kast",
        "first_kills",
        "first_deaths",
        "plants",
        "defuses",
        "econ_spend",
        "econ_value",
        "utility_used",
        "utility_damage",
        "won_match",
    ]
    for column in numeric_columns:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "agent_role" not in df.columns:
        df["agent_role"] = pd.NA
    if "agent" not in df.columns:
        df["agent"] = pd.NA
    df["agent"] = df["agent"].map(normalize_agent_name)
    df["agent_role"] = df["agent_role"].fillna(df["agent"].map(ROLE_MAP))
    df["kda_ratio_match"] = (df["kills"].fillna(0) + df["assists"].fillna(0)) / df["deaths"].replace(0, 1).fillna(1)
    df["entry_rate_match"] = _safe_divide(df["first_kills"], df["kills"])
    df["entry_success_rate_match"] = _safe_divide(df["first_kills"], df["first_kills"].fillna(0) + df["first_deaths"].fillna(0))
    df["support_score_match"] = (
        df["assists"].fillna(0)
        + df["plants"].fillna(0) * 0.75
        + df["defuses"].fillna(0) * 1.0
        + _safe_divide(df["utility_damage"], df["utility_used"]).fillna(0)
    )
    df["objective_score_match"] = df["plants"].fillna(0) + df["defuses"].fillna(0) * 1.25
    df["econ_efficiency_match"] = _safe_divide(df["damage"].fillna(df["adr"]), df["econ_spend"])
    df["utility_efficiency_match"] = _safe_divide(df["utility_damage"], df["utility_used"])
    df["damage_per_round_or_match"] = df["adr"].fillna(df["damage"])

    grouped = df.groupby(["cohort", "player_id", "player_name"], dropna=False)
    features = grouped.agg(
        matches_played=("match_id", "nunique"),
        win_rate=("won_match", "mean"),
        kda_ratio=("kda_ratio_match", "mean"),
        kills_per_match=("kills", "mean"),
        deaths_per_match=("deaths", "mean"),
        assists_per_match=("assists", "mean"),
        headshot_rate_mean=("headshot_rate", "mean"),
        damage_per_round_or_match=("damage_per_round_or_match", "mean"),
        acs_mean=("acs", "mean"),
        adr_mean=("adr", "mean"),
        kast_mean=("kast", "mean"),
        entry_rate=("entry_rate_match", "mean"),
        entry_success_rate=("entry_success_rate_match", "mean"),
        support_score=("support_score_match", "mean"),
        objective_score=("objective_score_match", "mean"),
        econ_efficiency=("econ_efficiency_match", "mean"),
        utility_efficiency=("utility_efficiency_match", "mean"),
    ).reset_index()

    volatility = grouped.agg(acs_std=("acs", "std"), adr_std=("adr", "std")).reset_index()
    features = features.merge(volatility, on=["cohort", "player_id", "player_name"], how="left")
    features["consistency_score"] = 1.0 / (1.0 + features["acs_std"].fillna(features["adr_std"]).fillna(0))

    role_entropy = grouped["agent_role"].apply(_entropy).reset_index(name="agent_role_entropy")
    role_primary_share = grouped["agent_role"].apply(_mode_share).reset_index(name="role_primary_share")
    primary_role = grouped["agent_role"].apply(_mode_value).reset_index(name="primary_role")
    primary_agent = grouped["agent"].apply(_mode_value).reset_index(name="primary_agent")
    map_entropy = grouped["match_id"].apply(lambda s: math.log2(max(1, s.nunique()))).reset_index(name="map_pool_entropy")

    features = features.merge(role_entropy, on=["cohort", "player_id", "player_name"], how="left")
    features = features.merge(role_primary_share, on=["cohort", "player_id", "player_name"], how="left")
    features = features.merge(primary_role, on=["cohort", "player_id", "player_name"], how="left")
    features = features.merge(primary_agent, on=["cohort", "player_id", "player_name"], how="left")
    features = features.merge(map_entropy, on=["cohort", "player_id", "player_name"], how="left")

    if "match_datetime" in df.columns and df["match_datetime"].notna().any():
        recent = df.sort_values("match_datetime").copy()
        recent["performance_proxy"] = recent["acs"].fillna(recent["adr"]).fillna(recent["kills"])
        recent_form = recent.groupby(["cohort", "player_id"])["performance_proxy"].agg(
            lambda x: x.tail(3).mean() - x.head(min(3, len(x))).mean()
        )
        features = features.merge(
            recent_form.reset_index(name="recent_form_delta"),
            on=["cohort", "player_id"],
            how="left",
        )
    else:
        features["recent_form_delta"] = np.nan

    features["clutch_proxy_rate"] = np.nan

    retained = [col for col in features.columns if col in {"cohort", "player_id", "player_name", "matches_played", "primary_role", "primary_agent"}]
    retained += [col for col in features.columns if features[col].notna().mean() >= min_feature_coverage and col not in retained]
    return features[retained]
