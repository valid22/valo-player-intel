from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from valo_player_intel.features.player_features import ROLE_MAP


@dataclass(slots=True)
class CohortClusteringResult:
    cohort: str
    assignments: pd.DataFrame
    profile_table: pd.DataFrame
    metrics: dict[str, float | int | str]
    pca_points: pd.DataFrame
    analysis_scope: str = "global"
    group_name: str | None = None


@dataclass(slots=True)
class AgentBehaviorResult:
    cohort: str
    agent_profiles: pd.DataFrame
    pca_points: pd.DataFrame
    metrics: dict[str, float | int | str]


ARCHETYPE_LIBRARY = {
    "entry": ["High-Tempo Entry", "Clinical Entry", "Space-Making Entry"],
    "carry": ["Aim-Heavy Carry", "Damage-Centric Carry", "Conversion Carry"],
    "support": ["Utility-Forward Support", "Objective-First Support", "Setup Support"],
    "anchor": ["Objective-Centric Anchor", "Low-Variance Anchor", "Stability Anchor"],
    "flex": ["Adaptive Flex", "Wide-Pool Flex", "Roaming Flex"],
    "specialist": ["Specialist Operator", "Role-Locked Specialist", "Niche Specialist"],
}

CLUSTER_FEATURE_PRIORITY = [
    "matches_played",
    "win_rate",
    "kda_ratio",
    "kills_per_match",
    "deaths_per_match",
    "assists_per_match",
    "headshot_rate_mean",
    "damage_per_round_or_match",
    "acs_mean",
    "adr_mean",
    "entry_rate",
    "entry_success_rate",
    "support_score",
    "objective_score",
    "consistency_score",
    "agent_role_entropy",
    "role_primary_share",
    "map_pool_entropy",
]


def _safe_value(row: pd.Series, column: str) -> float:
    value = row.get(column, 0.0)
    return 0.0 if pd.isna(value) else float(value)


def _ranked_features(z_row: pd.Series) -> tuple[list[str], list[str]]:
    numeric = z_row.drop(labels=["cluster_id"], errors="ignore").dropna()
    if numeric.empty:
        return [], []
    positive = numeric.sort_values(ascending=False).head(3).index.tolist()
    negative = numeric.sort_values(ascending=True).head(2).index.tolist()
    return positive, negative


def _feature_display_name(column: str) -> str:
    return column.replace("_", " ")


def _archetype_scores(z_series: pd.Series) -> dict[str, float]:
    return {
        "entry": _safe_value(z_series, "entry_rate") + 0.9 * _safe_value(z_series, "entry_success_rate") + 0.5 * _safe_value(z_series, "headshot_rate_mean"),
        "carry": _safe_value(z_series, "kda_ratio") + 0.8 * _safe_value(z_series, "damage_per_round_or_match") + 0.6 * _safe_value(z_series, "acs_mean") + 0.5 * _safe_value(z_series, "adr_mean"),
        "support": _safe_value(z_series, "support_score") + 0.9 * _safe_value(z_series, "objective_score") + 0.6 * _safe_value(z_series, "utility_efficiency"),
        "anchor": _safe_value(z_series, "consistency_score") + 0.5 * _safe_value(z_series, "win_rate") + 0.3 * _safe_value(z_series, "role_primary_share"),
        "flex": _safe_value(z_series, "agent_role_entropy") + 0.8 * _safe_value(z_series, "map_pool_entropy") + 0.3 * _safe_value(z_series, "matches_played"),
        "specialist": _safe_value(z_series, "role_primary_share") - 0.6 * _safe_value(z_series, "agent_role_entropy") + 0.5 * _safe_value(z_series, "headshot_rate_mean"),
    }


def _build_cluster_narratives(profiles: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [col for col in profiles.columns if col != "cluster_id" and pd.api.types.is_numeric_dtype(profiles[col])]
    z_profiles = profiles[["cluster_id"]].copy()
    for column in numeric_columns:
        series = profiles[column]
        std = series.std(ddof=0)
        z_profiles[column] = (series - series.mean()) / std if std and not np.isclose(std, 0.0) else 0.0

    used_labels: dict[str, int] = {}
    enriched_rows = []
    for row, z_row in zip(profiles.to_dict(orient="records"), z_profiles.to_dict(orient="records"), strict=False):
        z_series = pd.Series(z_row)
        scores = _archetype_scores(z_series)
        base_key = max(scores, key=scores.get)
        label_index = used_labels.get(base_key, 0)
        used_labels[base_key] = label_index + 1
        label_options = ARCHETYPE_LIBRARY[base_key]
        archetype_label = label_options[min(label_index, len(label_options) - 1)]

        standout_positive, standout_negative = _ranked_features(z_series)
        summary = "High on {} with lower {}.".format(
            ", ".join(_feature_display_name(name) for name in standout_positive[:2]) or "core traits",
            ", ".join(_feature_display_name(name) for name in standout_negative[:1]) or "secondary traits",
        )
        enriched_rows.append(
            {
                **row,
                "archetype_key": base_key,
                "archetype_label": archetype_label,
                "archetype_summary": summary,
                "standout_traits": ", ".join(_feature_display_name(name) for name in standout_positive),
                "suppressed_traits": ", ".join(_feature_display_name(name) for name in standout_negative),
            }
        )
    return pd.DataFrame(enriched_rows)


def _dominant_value(series: pd.Series) -> str | None:
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return None
    return str(cleaned.value_counts().idxmax())


def _top_distribution(series: pd.Series, top_n: int = 3) -> tuple[str, list[str]]:
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return "", []
    counts = cleaned.value_counts(normalize=True).head(top_n)
    text = ", ".join(f"{label} {share:.0%}" for label, share in counts.items())
    return text, [str(label) for label in counts.index.tolist()]


def _role_signature(cluster_df: pd.DataFrame) -> dict[str, object]:
    role_counts = (
        cluster_df["primary_role"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
    )
    if role_counts.empty:
        return {
            "dominant_role": None,
            "secondary_role": None,
            "dominant_role_share": None,
            "is_hybrid": False,
            "role_balance_type": "unknown",
            "top_roles": "",
            "top_agents": "",
            "top_role_1": None,
            "top_role_2": None,
            "top_role_3": None,
            "top_agent_1": None,
            "top_agent_2": None,
            "top_agent_3": None,
        }
    dominant_role = str(role_counts.index[0])
    dominant_share = float(role_counts.iloc[0] / role_counts.sum())
    secondary_role = str(role_counts.index[1]) if len(role_counts) > 1 else None
    if dominant_share >= 0.65:
        role_balance_type = "role_locked"
    elif dominant_share >= 0.42:
        role_balance_type = "role_leaning"
    else:
        role_balance_type = "cross_role"
    is_hybrid = role_balance_type != "role_locked" and secondary_role is not None
    top_roles_text, top_roles = _top_distribution(cluster_df["primary_role"])
    top_agents_text, top_agents = _top_distribution(cluster_df["primary_agent"])
    return {
        "dominant_role": dominant_role,
        "secondary_role": secondary_role,
        "dominant_role_share": dominant_share,
        "is_hybrid": is_hybrid,
        "role_balance_type": role_balance_type,
        "top_roles": top_roles_text,
        "top_agents": top_agents_text,
        "top_role_1": top_roles[0] if len(top_roles) > 0 else None,
        "top_role_2": top_roles[1] if len(top_roles) > 1 else None,
        "top_role_3": top_roles[2] if len(top_roles) > 2 else None,
        "top_agent_1": top_agents[0] if len(top_agents) > 0 else None,
        "top_agent_2": top_agents[1] if len(top_agents) > 1 else None,
        "top_agent_3": top_agents[2] if len(top_agents) > 2 else None,
    }


def _role_flavored_label(
    base_label: str,
    primary_role: str | None,
    role_balance_type: str | None,
    secondary_role: str | None = None,
    analysis_scope: str = "global",
) -> str:
    role_prefix = {
        "Duelist": {
            "High-Tempo Entry": "Duelist Entry",
            "Clinical Entry": "Duelist Finisher",
            "Aim-Heavy Carry": "Duelist Carry",
            "Space-Making Entry": "Space-Making Duelist",
        },
        "Controller": {
            "Objective-Centric Anchor": "Controller Anchor",
            "Utility-Forward Support": "Controller Support",
            "Adaptive Flex": "Controller Flex",
        },
        "Initiator": {
            "Utility-Forward Support": "Initiator Setup",
            "Adaptive Flex": "Initiator Flex",
            "High-Tempo Entry": "Initiator Entry",
        },
        "Sentinel": {
            "Objective-Centric Anchor": "Sentinel Anchor",
            "Utility-Forward Support": "Sentinel Support",
            "Specialist Operator": "Sentinel Specialist",
        },
    }
    flavored = role_prefix.get(primary_role or "", {}).get(base_label, base_label)
    if analysis_scope == "role" and primary_role:
        return flavored
    if role_balance_type == "cross_role":
        if base_label == "Role-Locked Specialist":
            return "Cross-Role Specialist"
        return f"Cross-Role {base_label}"
    if role_balance_type == "role_leaning" and primary_role and secondary_role:
        return f"{primary_role}-Leaning {flavored}"
    return flavored


def _cluster_summary_with_role(row: dict, analysis_scope: str) -> str:
    role = row.get("dominant_role")
    secondary_role = row.get("secondary_role")
    dominant_role_share = row.get("dominant_role_share")
    role_balance_type = row.get("role_balance_type")
    top_roles = row.get("top_roles")
    top_agents = row.get("top_agents")
    base = row.get("archetype_summary", "")
    fragments = []
    if analysis_scope == "role" and role:
        fragments.append(f"This role-specific cluster sits inside the {role.lower()} pool")
    elif role and role_balance_type == "cross_role" and secondary_role:
        fragments.append("This cluster is behaviorally mixed across multiple roles rather than role-locked")
    elif role and role_balance_type == "role_leaning" and secondary_role:
        share_suffix = f" ({dominant_role_share:.0%} of players)" if isinstance(dominant_role_share, (int, float)) else ""
        fragments.append(f"This cluster leans {role.lower()}{share_suffix} with a meaningful {secondary_role.lower()} secondary lane")
    elif role:
        share_suffix = f" ({dominant_role_share:.0%} of players)" if isinstance(dominant_role_share, (int, float)) else ""
        fragments.append(f"This cluster is mostly {role.lower()}{share_suffix}")
    if top_roles:
        fragments.append(f"Top roles: {top_roles}")
    if top_agents:
        fragments.append(f"Top agents: {top_agents}")
    prefix = ", ".join(fragments)
    if prefix and base:
        return f"{prefix}. {base}"
    return prefix or base


def _cluster_frame(
    cohort_df: pd.DataFrame,
    cohort: str,
    cluster_candidates: tuple[int, ...],
    default_cluster_count: int,
    random_state: int,
    analysis_scope: str,
    group_name: str | None = None,
) -> CohortClusteringResult | None:
    id_columns = ["cohort", "player_id", "player_name"]
    metadata_columns = ["primary_role", "primary_agent"]
    numeric_columns = [col for col in CLUSTER_FEATURE_PRIORITY if col in cohort_df.columns and pd.api.types.is_numeric_dtype(cohort_df[col]) and not cohort_df[col].isna().all()]
    if len(cohort_df) < 8 or len(numeric_columns) < 4:
        return None

    X = cohort_df[numeric_columns]
    prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    transformed = prep.fit_transform(X)

    best_score = -1.0
    best_labels = None
    best_k = default_cluster_count
    for k in cluster_candidates:
        if len(cohort_df) <= k:
            continue
        labels = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit_predict(transformed)
        score = silhouette_score(transformed, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k
    if best_labels is None:
        best_k = min(default_cluster_count, max(2, len(cohort_df) // 2))
        best_labels = KMeans(n_clusters=best_k, n_init=10, random_state=random_state).fit_predict(transformed)

    gmm = GaussianMixture(n_components=best_k, random_state=random_state)
    dbscan = DBSCAN(eps=1.5, min_samples=max(4, min(10, len(cohort_df) // 15)))
    gmm.fit(transformed)
    dbscan.fit(transformed)

    assignments = cohort_df[id_columns].copy()
    assignments["cluster_id"] = best_labels
    assignments["analysis_scope"] = analysis_scope
    assignments["group_name"] = group_name

    profile_base = cohort_df.assign(cluster_id=best_labels).groupby("cluster_id")[numeric_columns].mean().reset_index()
    cluster_sizes = assignments.groupby("cluster_id").size().reset_index(name="cluster_size")
    cluster_members = cohort_df.assign(cluster_id=best_labels)
    role_signals = pd.DataFrame(
        [{"cluster_id": cluster_id, **_role_signature(cluster_df)} for cluster_id, cluster_df in cluster_members.groupby("cluster_id")]
    )
    profiles = _build_cluster_narratives(profile_base).merge(cluster_sizes, on="cluster_id", how="left").merge(role_signals, on="cluster_id", how="left")
    profiles["analysis_scope"] = analysis_scope
    profiles["group_name"] = group_name
    profiles["archetype_label"] = profiles.apply(
        lambda row: _role_flavored_label(
            row["archetype_label"],
            row.get("dominant_role"),
            row.get("role_balance_type"),
            row.get("secondary_role"),
            analysis_scope=analysis_scope,
        ),
        axis=1,
    )
    profiles["archetype_summary"] = profiles.apply(lambda row: _cluster_summary_with_role(row.to_dict(), analysis_scope), axis=1)
    assignments = assignments.merge(
        profiles[["cluster_id", "analysis_scope", "group_name", "archetype_label", "archetype_summary"]],
        on=["cluster_id", "analysis_scope", "group_name"],
        how="left",
    )

    pca = PCA(n_components=2, random_state=random_state)
    pca_points = pd.DataFrame(pca.fit_transform(transformed), columns=["pca_1", "pca_2"])
    pca_points = pd.concat([assignments, pca_points], axis=1)

    metrics = {
        "cohort": cohort,
        "analysis_scope": analysis_scope,
        "group_name": group_name or cohort,
        "selected_cluster_count": int(best_k),
        "silhouette_score": float(best_score),
        "gmm_bic": float(gmm.bic(transformed)),
        "dbscan_clusters": int(len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)),
    }
    return CohortClusteringResult(
        cohort=cohort,
        assignments=assignments,
        profile_table=profiles,
        metrics=metrics,
        pca_points=pca_points,
        analysis_scope=analysis_scope,
        group_name=group_name,
    )


def run_clustering(
    player_features: pd.DataFrame,
    cluster_candidates: tuple[int, ...],
    default_cluster_count: int,
    random_state: int,
) -> list[CohortClusteringResult]:
    results: list[CohortClusteringResult] = []
    for cohort, cohort_df in player_features.groupby("cohort"):
        result = _cluster_frame(
            cohort_df,
            cohort=cohort,
            cluster_candidates=cluster_candidates,
            default_cluster_count=default_cluster_count,
            random_state=random_state,
            analysis_scope="global",
            group_name=None,
        )
        if result is not None:
            results.append(result)
    return results


def run_role_conditioned_clustering(
    player_features: pd.DataFrame,
    cluster_candidates: tuple[int, ...],
    default_cluster_count: int,
    random_state: int,
) -> list[CohortClusteringResult]:
    role_results: list[CohortClusteringResult] = []
    for cohort, cohort_df in player_features.groupby("cohort"):
        for role, role_df in cohort_df.groupby("primary_role"):
            if pd.isna(role) or len(role_df) < 25:
                continue
            role_candidates = tuple(k for k in cluster_candidates if k <= max(4, len(role_df) // 12))
            if not role_candidates:
                role_candidates = (3, 4)
            result = _cluster_frame(
                role_df,
                cohort=cohort,
                cluster_candidates=role_candidates,
                default_cluster_count=min(default_cluster_count, 4),
                random_state=random_state,
                analysis_scope="role",
                group_name=str(role),
            )
            if result is not None:
                role_results.append(result)
    return role_results


def summarize_cluster_outcomes(player_features: pd.DataFrame, global_results: list[CohortClusteringResult]) -> pd.DataFrame:
    rows = []
    for result in global_results:
        merged = result.assignments.merge(player_features, on=["cohort", "player_id", "player_name"], how="left")
        summary = (
            merged.groupby(["cohort", "cluster_id", "archetype_label"], dropna=False)
            .agg(
                players=("player_id", "nunique"),
                avg_win_rate=("win_rate", "mean"),
                avg_kda_ratio=("kda_ratio", "mean"),
                avg_matches_played=("matches_played", "mean"),
                avg_role_entropy=("agent_role_entropy", "mean"),
            )
            .reset_index()
        )
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compare_cohorts(global_results: list[CohortClusteringResult]) -> pd.DataFrame:
    rows = []
    for result in global_results:
        profile = result.profile_table.copy()
        profile["cohort"] = result.cohort
        rows.append(profile)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if combined.empty:
        return combined
    metrics = [col for col in ["cluster_size", "win_rate", "kda_ratio", "support_score", "objective_score", "consistency_score"] if col in combined.columns]
    comparison_rows = []
    for metric in metrics:
        grouped = combined.groupby("cohort")[metric].mean()
        comparison_rows.append(
            {
                "metric": metric,
                "pro_mean": float(grouped.get("pro", np.nan)),
                "public_mean": float(grouped.get("public", np.nan)),
                "delta_pro_minus_public": float(grouped.get("pro", np.nan) - grouped.get("public", np.nan)),
            }
        )
    return pd.DataFrame(comparison_rows)


def analyze_agent_behavior(
    player_features: pd.DataFrame,
    random_state: int,
) -> list[AgentBehaviorResult]:
    feature_columns = [col for col in CLUSTER_FEATURE_PRIORITY if col in player_features.columns]
    min_agent_players = 20
    results: list[AgentBehaviorResult] = []
    for cohort, cohort_df in player_features.groupby("cohort"):
        agent_df = (
            cohort_df.dropna(subset=["primary_agent"])
            .groupby("primary_agent")[feature_columns]
            .mean()
            .reset_index()
            .rename(columns={"primary_agent": "agent"})
        )
        counts = cohort_df["primary_agent"].value_counts().rename_axis("agent").reset_index(name="player_count")
        agent_df = agent_df.merge(counts, on="agent", how="left")
        agent_df["sample_tier"] = np.where(agent_df["player_count"] >= min_agent_players, "stable", "low_sample")
        stable_agent_df = agent_df[agent_df["player_count"] >= min_agent_players].copy()
        if len(stable_agent_df) < 4:
            continue

        nominal_roles = pd.Series(stable_agent_df["agent"]).map(ROLE_MAP)
        role_prototypes = cohort_df.dropna(subset=["primary_role"]).groupby("primary_role")[feature_columns].mean()

        prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        transformed = prep.fit_transform(stable_agent_df[feature_columns])
        role_proto_transformed = prep.transform(role_prototypes[feature_columns]) if not role_prototypes.empty else np.empty((0, len(feature_columns)))
        k = min(4, max(3, len(stable_agent_df) // 5)) if len(stable_agent_df) >= 6 else min(3, len(stable_agent_df))
        labels = KMeans(n_clusters=max(2, k), n_init=20, random_state=random_state).fit_predict(transformed)

        if len(role_prototypes.index) > 0:
            inferred_roles = []
            confidences = []
            for row in transformed:
                distances = ((role_proto_transformed - row) ** 2).sum(axis=1)
                order = np.argsort(distances)
                best_idx = int(order[0])
                inferred_roles.append(str(role_prototypes.index[best_idx]))
                if len(order) > 1 and distances[order[1]] > 0:
                    confidences.append(float(1 - (distances[best_idx] / distances[order[1]])))
                else:
                    confidences.append(1.0)
        else:
            inferred_roles = [None] * len(agent_df)
            confidences = [np.nan] * len(agent_df)

        pca = PCA(n_components=2, random_state=random_state)
        pca_points = pd.DataFrame(pca.fit_transform(transformed), columns=["pca_1", "pca_2"])
        pca_points["agent"] = stable_agent_df["agent"]
        pca_points["cluster_id"] = labels
        pca_points["cohort"] = cohort

        role_cluster_names = {}
        z_agent_df = stable_agent_df[feature_columns].copy()
        for column in feature_columns:
            std = z_agent_df[column].std(ddof=0)
            z_agent_df[column] = (z_agent_df[column] - z_agent_df[column].mean()) / std if std and not np.isclose(std, 0.0) else 0.0
        for cluster_id in sorted(set(labels)):
            cluster_slice = z_agent_df.loc[labels == cluster_id, feature_columns].mean()
            cluster_scores = _archetype_scores(cluster_slice)
            role_cluster_names[cluster_id] = max(cluster_scores, key=cluster_scores.get)
        profile = stable_agent_df.copy()
        profile["cohort"] = cohort
        profile["cluster_id"] = labels
        profile["nominal_role"] = nominal_roles
        profile["inferred_behavior_role"] = inferred_roles
        profile["inference_confidence"] = confidences
        profile["role_alignment"] = np.where(profile["nominal_role"] == profile["inferred_behavior_role"], "aligned", "shifted")
        profile["alignment_stability"] = np.where(profile["inference_confidence"] >= 0.15, "stable", "uncertain")
        profile["role_alignment"] = np.where(profile["nominal_role"] == profile["inferred_behavior_role"], "aligned", "shifted")
        profile["agent_behavior_archetype"] = profile["cluster_id"].map(role_cluster_names)
        profile["analysis_scope"] = "agent_behavior"
        low_sample_agents = agent_df[agent_df["player_count"] < min_agent_players].copy()
        if not low_sample_agents.empty:
            low_sample_agents["cohort"] = cohort
            low_sample_agents["cluster_id"] = pd.NA
            low_sample_agents["nominal_role"] = low_sample_agents["agent"].map(ROLE_MAP)
            low_sample_agents["inferred_behavior_role"] = pd.NA
            low_sample_agents["inference_confidence"] = np.nan
            low_sample_agents["role_alignment"] = "insufficient_sample"
            low_sample_agents["alignment_stability"] = "low_sample"
            low_sample_agents["agent_behavior_archetype"] = "insufficient_sample"
            low_sample_agents["analysis_scope"] = "agent_behavior"
            profile = pd.concat([profile, low_sample_agents], ignore_index=True, sort=False)
        metrics = {
            "cohort": cohort,
            "agent_count": int(len(profile)),
            "agent_clusters": int(len(set(labels))),
            "alignment_rate": float((profile["role_alignment"] == "aligned").mean()),
            "stable_alignment_rate": float(
                (
                    profile.loc[profile["alignment_stability"] == "stable", "role_alignment"]
                    .eq("aligned")
                    .mean()
                )
            ) if (profile["alignment_stability"] == "stable").any() else np.nan,
            "stable_agent_count": int((profile["alignment_stability"] == "stable").sum()),
            "low_sample_agent_count": int((profile["alignment_stability"] == "low_sample").sum()),
            "min_agent_players": int(min_agent_players),
        }
        results.append(AgentBehaviorResult(cohort=cohort, agent_profiles=profile, pca_points=pca_points, metrics=metrics))
    return results
