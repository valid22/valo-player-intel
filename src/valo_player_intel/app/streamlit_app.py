from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from valo_player_intel.data.io import read_table


REPORT_PAGES = ["Overview", "Segmentation", "Modeling", "Comparison", "Data"]
COHORT_COLORS = {"pro": "#ff6b3d", "public": "#52d1b5"}
ROLE_COLORS = {"Duelist": "#ff7f50", "Controller": "#6bb8ff", "Initiator": "#9f86ff", "Sentinel": "#66d9a3"}
AGENT_RADAR_COLUMNS = ["win_rate", "kda_ratio", "entry_rate", "support_score", "objective_score", "consistency_score"]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_optional_table(path: Path) -> pd.DataFrame:
    return read_table(path) if path.exists() else pd.DataFrame()


def _metric(value: float | int | None, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if pct:
        return f"{float(value):.1%}"
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)


def _section(eyebrow: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, rgba(24,28,37,0.97), rgba(12,14,20,0.97));border:1px solid rgba(255,255,255,0.08);border-radius:24px;padding:1.2rem 1.3rem;margin-bottom:1rem;box-shadow:0 20px 50px rgba(0,0,0,0.28);">
            <div style="font-size:0.78rem;letter-spacing:0.12em;text-transform:uppercase;color:#8ea4ff;">{eyebrow}</div>
            <div style="font-size:1.65rem;font-weight:700;color:#f4f7fb;margin-top:0.35rem;">{title}</div>
            <div style="font-size:0.97rem;color:#b8c0d0;margin-top:0.55rem;line-height:1.6;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div style="background:linear-gradient(180deg, rgba(22,24,31,0.96), rgba(11,13,19,0.96));border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:1rem;min-height:122px;">
            <div style="font-size:0.78rem;letter-spacing:0.08em;text-transform:uppercase;color:#96a0b5;">{label}</div>
            <div style="font-size:1.85rem;font-weight:700;color:#f4f7fb;margin-top:0.35rem;">{value}</div>
            <div style="font-size:0.88rem;color:#9ba8bf;margin-top:0.55rem;line-height:1.45;">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _story(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="background:rgba(17,19,25,0.94);border:1px solid rgba(255,255,255,0.08);border-radius:18px;padding:1rem 1.05rem;margin-bottom:0.8rem;">
            <div style="font-size:0.82rem;letter-spacing:0.08em;text-transform:uppercase;color:#8ea4ff;">{title}</div>
            <div style="font-size:0.94rem;color:#c7d0df;line-height:1.6;margin-top:0.45rem;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _cluster_card(row: dict) -> None:
    badge = ""
    if bool(row.get("is_hybrid")):
        badge = '<span style="font-size:0.72rem;padding:0.26rem 0.5rem;border-radius:999px;background:rgba(82,209,181,0.16);color:#52d1b5;border:1px solid rgba(82,209,181,0.35);">Hybrid</span>'
    st.markdown(
        f"""
        <div style="background:rgba(17,19,25,0.94);border:1px solid rgba(255,255,255,0.08);border-radius:22px;padding:1rem 1.05rem;margin-bottom:0.8rem;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">
                <div>
                    <div style="font-size:0.77rem;text-transform:uppercase;letter-spacing:0.1em;color:#90a0bf;">Cluster {int(row['cluster_id'])}</div>
                    <div style="font-size:1.22rem;font-weight:700;color:#f3f6fb;margin-top:0.25rem;">{row.get('archetype_label', 'n/a')}</div>
                </div>
                <div style="display:flex;align-items:center;gap:0.5rem;">{badge}<div style="font-size:0.92rem;color:#dbe3f0;">{int(row.get('cluster_size', 0))} players</div></div>
            </div>
            <div style="font-size:0.93rem;color:#c5cede;line-height:1.55;margin-top:0.55rem;">{row.get('archetype_summary', '')}</div>
            <div style="font-size:0.86rem;color:#b9c4d8;margin-top:0.6rem;"><strong>Role mix</strong> {row.get('top_roles', 'n/a')}</div>
            <div style="font-size:0.86rem;color:#7fd7c2;margin-top:0.2rem;"><strong>Top agents</strong> {row.get('top_agents', 'n/a')}</div>
            <div style="font-size:0.86rem;color:#8ea4ff;margin-top:0.55rem;"><strong>Standout</strong> {row.get('standout_traits', 'n/a')}</div>
            <div style="font-size:0.86rem;color:#98a6bb;margin-top:0.2rem;"><strong>Lower signal</strong> {row.get('suppressed_traits', 'n/a')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_cluster_grid(profiles: pd.DataFrame, columns: int = 2) -> None:
    rows = profiles.sort_values("cluster_id").to_dict(orient="records")
    for start in range(0, len(rows), columns):
        grid = st.columns(columns)
        for idx, row in enumerate(rows[start : start + columns]):
            with grid[idx]:
                _cluster_card(row)


def _agent_role_radar(profiles: pd.DataFrame, agent_name: str) -> go.Figure:
    columns = [column for column in AGENT_RADAR_COLUMNS if column in profiles.columns]
    if not columns:
        return go.Figure()
    working = profiles.copy()
    normalized = working[["agent", "nominal_role", "inferred_behavior_role", *columns]].copy()
    for column in columns:
        series = normalized[column]
        span = series.max() - series.min()
        normalized[column] = 0.5 if pd.isna(span) or span == 0 else (series - series.min()) / span

    agent_row = normalized.loc[normalized["agent"] == agent_name].iloc[0]
    nominal_role = agent_row["nominal_role"]
    inferred_role = agent_row["inferred_behavior_role"]
    nominal_profile = normalized.loc[normalized["nominal_role"] == nominal_role, columns].mean()
    inferred_profile = normalized.loc[normalized["inferred_behavior_role"] == inferred_role, columns].mean() if pd.notna(inferred_role) else None
    theta = [column.replace("_", " ") for column in columns] + [columns[0].replace("_", " ")]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=agent_row[columns].tolist() + [agent_row[columns].tolist()[0]],
            theta=theta,
            fill="toself",
            name=agent_name,
            line={"color": "#ff6b3d", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=nominal_profile.tolist() + [nominal_profile.tolist()[0]],
            theta=theta,
            name=f"Nominal {nominal_role}",
            line={"color": "#52d1b5", "width": 2, "dash": "dash"},
        )
    )
    if inferred_profile is not None and not inferred_profile.empty:
        fig.add_trace(
            go.Scatterpolar(
                r=inferred_profile.tolist() + [inferred_profile.tolist()[0]],
                theta=theta,
                name=f"Inferred {inferred_role}",
                line={"color": "#8ea4ff", "width": 2},
            )
        )
    fig.update_layout(
        title=f"{agent_name} vs role prototypes",
        polar={"radialaxis": {"visible": True, "range": [0, 1], "gridcolor": "rgba(255,255,255,0.12)"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dbe3f0"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def _load_global_bundle(project_root: Path, cohort: str) -> dict[str, pd.DataFrame]:
    return {
        "profiles": _load_optional_table(project_root / "artifacts" / "segmentation" / f"{cohort}_cluster_profiles.parquet"),
        "assignments": _load_optional_table(project_root / "artifacts" / "segmentation" / f"{cohort}_clusters.parquet"),
        "pca": _load_optional_table(project_root / "artifacts" / "segmentation" / f"{cohort}_pca_points.parquet"),
    }


def _load_role_bundle(project_root: Path, cohort: str, role: str) -> dict[str, pd.DataFrame]:
    slug = role.lower().replace("/", "_").replace(" ", "_")
    return {
        "profiles": _load_optional_table(project_root / "artifacts" / "segmentation" / f"{cohort}_{slug}_role_cluster_profiles.parquet"),
        "assignments": _load_optional_table(project_root / "artifacts" / "segmentation" / f"{cohort}_{slug}_role_clusters.parquet"),
        "pca": _load_optional_table(project_root / "artifacts" / "segmentation" / f"{cohort}_{slug}_role_pca_points.parquet"),
    }


def _load_agent_bundle(project_root: Path, cohort: str) -> dict[str, pd.DataFrame]:
    return {
        "profiles": _load_optional_table(project_root / "artifacts" / "segmentation" / f"{cohort}_agent_behavior_profiles.parquet"),
        "pca": _load_optional_table(project_root / "artifacts" / "segmentation" / f"{cohort}_agent_behavior_pca_points.parquet"),
    }


def _timeline_frame(project_root: Path, assignments: pd.DataFrame, scope_group: str | None = None) -> pd.DataFrame:
    player_matches = _load_optional_table(project_root / "data" / "interim" / "player_matches.parquet")
    matches = _load_optional_table(project_root / "data" / "interim" / "matches.parquet")
    if assignments.empty or player_matches.empty or matches.empty:
        return pd.DataFrame()
    cols = ["cohort", "player_id", "archetype_label"]
    if "group_name" in assignments.columns:
        cols.append("group_name")
    timeline = player_matches.merge(assignments[cols].drop_duplicates(), on=["cohort", "player_id"], how="inner")
    timeline = timeline.merge(matches[["match_id", "match_datetime", "cohort"]], on=["match_id", "cohort"], how="left")
    if scope_group is not None and "group_name" in timeline.columns:
        timeline = timeline[timeline["group_name"] == scope_group]
    if timeline.empty or "match_datetime" not in timeline.columns or timeline["match_datetime"].isna().all():
        return pd.DataFrame()
    timeline["period"] = pd.to_datetime(timeline["match_datetime"], utc=True, errors="coerce").dt.floor("D")
    timeline = timeline.dropna(subset=["period"])
    prevalence = timeline.groupby(["period", "archetype_label"]).size().reset_index(name="appearances")
    totals = prevalence.groupby("period")["appearances"].transform("sum")
    prevalence["share"] = prevalence["appearances"] / totals
    return prevalence


def _role_files(project_root: Path, cohort: str) -> list[str]:
    roles = []
    for path in (project_root / "artifacts" / "segmentation").glob(f"{cohort}_*_role_cluster_profiles.parquet"):
        name = path.name.removeprefix(f"{cohort}_").removesuffix("_role_cluster_profiles.parquet")
        roles.append(name.replace("_", " ").title())
    return sorted(set(roles))


def _render_overview(project_root: Path, results: dict) -> None:
    raw_vct = _load_json(project_root / "data" / "raw" / "vct" / "vlr_matches.json").get("matches", [])
    raw_public = _load_json(project_root / "data" / "raw" / "public" / "henrik_matches.json").get("matches", [])
    player_features = _load_optional_table(project_root / "data" / "processed" / "player_features.parquet")
    best_models = results.get("prediction", {}).get("best_models", {})
    _section(
        "VALORANT Intelligence Report",
        "Behavioral archetypes plus win modeling on real pro and public data",
        "This report now separates the problem into three layers: global behavior clusters, role-conditioned archetypes, and agent-behavior analysis. The supervised side measures whether these structures also carry predictive signal for match outcomes.",
    )
    cols = st.columns(4)
    with cols[0]:
        _card("Pro Matches", str(len(raw_vct)), "Real VCT / official-play matches in the raw dataset.")
    with cols[1]:
        _card("Public Matches", str(len(raw_public)), "Real public competitive matches gathered from Henrik.")
    with cols[2]:
        _card("Pro Best Model", str(best_models.get("pro", {}).get("model_name", "n/a")), f"ROC AUC {_metric(best_models.get('pro', {}).get('roc_auc'))}")
    with cols[3]:
        _card("Public Best Model", str(best_models.get("public", {}).get("model_name", "n/a")), f"ROC AUC {_metric(best_models.get('public', {}).get('roc_auc'))}")

    if not player_features.empty:
        counts = player_features.groupby(["cohort", "primary_role"]).size().reset_index(name="players")
        fig = px.bar(
            counts,
            x="cohort",
            y="players",
            color="primary_role",
            title="Player coverage by cohort and primary role",
            barmode="stack",
            color_discrete_map=ROLE_COLORS,
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        pro_seg = results.get("segmentation", {}).get("pro", {})
        public_seg = results.get("segmentation", {}).get("public", {})
        _story(
            "Unsupervised headline",
            f"Global clustering separates pro players at silhouette <strong>{_metric(pro_seg.get('silhouette_score'))}</strong> and public players at <strong>{_metric(public_seg.get('silhouette_score'))}</strong>. The role-specific layer is where the analysis becomes more VALORANT-native.",
        )
    with right:
        _story(
            "Supervised headline",
            f"The best public model improves Brier error by <strong>{_metric(best_models.get('public', {}).get('brier_improvement_vs_baseline'), pct=True)}</strong> versus the baseline. The best pro model improves by <strong>{_metric(best_models.get('pro', {}).get('brier_improvement_vs_baseline'), pct=True)}</strong>.",
        )

    with st.expander("Raw run summary", expanded=False):
        st.json(results)


def _global_segmentation_view(project_root: Path, results: dict) -> None:
    cohort = st.segmented_control("Cohort", ["pro", "public"], default="pro", key="global_cohort")
    bundle = _load_global_bundle(project_root, cohort)
    profiles, assignments, pca = bundle["profiles"], bundle["assignments"], bundle["pca"]
    if profiles.empty:
        st.info("Global segmentation artifacts not found.")
        return
    silhouette = results.get("segmentation", {}).get(cohort, {}).get("silhouette_score")
    _section(
        "Global Unsupervised Learnings",
        f"{cohort.title()} behavioral clusters",
        f"These are broad behavioral clusters across all players in the cohort. They are useful for exploratory style discovery, but many are intentionally cross-role because they cluster behavior, not draft position. Silhouette: <strong>{_metric(silhouette)}</strong>.",
    )
    row_a, row_b = st.columns(2)
    with row_a:
        if not pca.empty:
            fig = px.scatter(pca, x="pca_1", y="pca_2", color="archetype_label", hover_data=["player_name", "cluster_id"], title=f"{cohort.title()} global archetype map", color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
            st.plotly_chart(fig, use_container_width=True)
    with row_b:
        heatmap_cols = [c for c in ["win_rate", "kda_ratio", "entry_rate", "support_score", "objective_score", "consistency_score"] if c in profiles.columns]
        if heatmap_cols:
            heat = go.Figure(
                data=go.Heatmap(
                    z=profiles[heatmap_cols].to_numpy(),
                    x=[c.replace("_", " ") for c in heatmap_cols],
                    y=profiles["archetype_label"],
                    colorscale=[[0, "#1f3145"], [0.5, "#141922"], [1, "#ff7c4d"]],
                )
            )
            heat.update_layout(title="Cluster trait heatmap", paper_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
            st.plotly_chart(heat, use_container_width=True)

    timeline = _timeline_frame(project_root, assignments)
    if not timeline.empty:
        line = px.area(
            timeline,
            x="period",
            y="share",
            color="archetype_label",
            title="Global archetype prevalence over time",
            groupnorm="fraction",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"}, yaxis_tickformat=".0%")
        st.plotly_chart(line, use_container_width=True)
        st.caption("This timeline shows participation share over time using the current archetype assignment for each player. It is descriptive, not a historical re-clustering.")

    _render_cluster_grid(profiles, columns=2)

    with st.expander("Global cluster profiles", expanded=False):
        st.dataframe(profiles, use_container_width=True)


def _role_segmentation_view(project_root: Path, results: dict) -> None:
    cohort = st.segmented_control("Role cohort", ["pro", "public"], default="pro", key="role_cohort")
    roles = _role_files(project_root, cohort)
    if not roles:
        st.info("Role-conditioned artifacts not found.")
        return
    role = st.selectbox("Role pool", roles, key="role_pool")
    bundle = _load_role_bundle(project_root, cohort, role)
    profiles, assignments, pca = bundle["profiles"], bundle["assignments"], bundle["pca"]
    if profiles.empty:
        st.info("Role-conditioned profiles not found.")
        return
    role_summary = results.get("role_segmentation", {}).get(cohort, {}).get(role, {})
    _section(
        "Role-Conditioned Learnings",
        f"{cohort.title()} {role} archetypes",
        f"This is the higher-value segmentation layer. Instead of mixing all players together, it clusters only the {role.lower()} pool. That makes the archetypes much closer to real VALORANT role usage. Silhouette: <strong>{_metric(role_summary.get('silhouette_score'))}</strong>.",
    )
    row_a, row_b = st.columns(2)
    with row_a:
        if not pca.empty:
            fig = px.scatter(pca, x="pca_1", y="pca_2", color="archetype_label", hover_data=["player_name", "cluster_id"], title=f"{cohort.title()} {role} archetype map", color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
            st.plotly_chart(fig, use_container_width=True)
    with row_b:
        share = px.pie(profiles, names="archetype_label", values="cluster_size", title="Role archetype share", hole=0.48, color_discrete_sequence=px.colors.qualitative.Set2)
        share.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
        st.plotly_chart(share, use_container_width=True)
    timeline = _timeline_frame(project_root, assignments, scope_group=role)
    if not timeline.empty:
        line = px.line(
            timeline,
            x="period",
            y="share",
            color="archetype_label",
            markers=True,
            title=f"{role} archetype prevalence over time",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"}, yaxis_tickformat=".0%")
        st.plotly_chart(line, use_container_width=True)
        st.caption("This timeline is strongest for the public cohort because public matches currently have complete timestamps.")
    _render_cluster_grid(profiles, columns=2)
    with st.expander("Role-conditioned profiles", expanded=False):
        st.dataframe(profiles, use_container_width=True)


def _agent_behavior_view(project_root: Path, results: dict) -> None:
    cohort = st.segmented_control("Agent cohort", ["pro", "public"], default="pro", key="agent_cohort")
    bundle = _load_agent_bundle(project_root, cohort)
    profiles, pca = bundle["profiles"], bundle["pca"]
    if profiles.empty:
        st.info("Agent behavior artifacts not found.")
        return
    behavior_metrics = results.get("agent_behavior", {}).get(cohort, {})
    alignment = behavior_metrics.get("alignment_rate")
    stable_alignment = behavior_metrics.get("stable_alignment_rate")
    stable_agents = behavior_metrics.get("stable_agent_count")
    low_sample_agents = behavior_metrics.get("low_sample_agent_count")
    _section(
        "Agent Behavior Analysis",
        f"{cohort.title()} agents clustered by real usage",
        f"This layer asks whether agents are actually played the way their nominal class suggests. Current nominal-role alignment rate: <strong>{_metric(alignment, pct=True)}</strong>. Among stable agent profiles, alignment is <strong>{_metric(stable_alignment, pct=True)}</strong> across <strong>{stable_agents or 0}</strong> agents. <strong>{low_sample_agents or 0}</strong> agents are shown as insufficient-sample rather than being hidden.",
    )
    left, right = st.columns([1.2, 1.8])
    with left:
        aligned = profiles["role_alignment"].value_counts(normalize=True)
        _story(
            "Agent usage finding",
            f"In the {cohort} cohort, <strong>{_metric(aligned.get('aligned', 0.0), pct=True)}</strong> of represented agent profiles stay aligned with their nominal role and <strong>{_metric(aligned.get('shifted', 0.0), pct=True)}</strong> show behavioral drift. Low-sample agents remain in the table as insufficient evidence instead of disappearing. Use the heatmap and confidence plot together: the heatmap shows where agents land, while the confidence plot shows how much to trust each assignment.",
        )
        st.dataframe(
            profiles[
                [
                    "agent",
                    "nominal_role",
                    "inferred_behavior_role",
                    "role_alignment",
                    "alignment_stability",
                    "inference_confidence",
                    "agent_behavior_archetype",
                    "player_count",
                ]
            ].sort_values(["alignment_stability", "player_count"], ascending=[True, False]),
            use_container_width=True,
        )
    with right:
        if not pca.empty:
            fig = px.scatter(
                pca.merge(profiles[["agent", "nominal_role", "inferred_behavior_role", "role_alignment"]], on="agent", how="left"),
                x="pca_1",
                y="pca_2",
                color="role_alignment",
                symbol="nominal_role",
                hover_data=["agent", "inferred_behavior_role"],
                title=f"{cohort.title()} agent behavior map",
                color_discrete_map={"aligned": "#52d1b5", "shifted": "#ff6b3d"},
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
            st.plotly_chart(fig, use_container_width=True)
    row_a, row_b = st.columns(2)
    with row_a:
        heatmap_frame = profiles[["nominal_role", "inferred_behavior_role", "player_count"]].dropna()
        if not heatmap_frame.empty:
            role_heat = heatmap_frame.groupby(["nominal_role", "inferred_behavior_role"])["player_count"].sum().reset_index()
            fig = px.density_heatmap(
                role_heat,
                x="nominal_role",
                y="inferred_behavior_role",
                z="player_count",
                text_auto=True,
                color_continuous_scale="Tealgrn",
                title="Nominal role vs inferred behavior role",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
            st.plotly_chart(fig, use_container_width=True)
    with row_b:
        confidence_fig = px.scatter(
            profiles.sort_values("player_count", ascending=False),
            x="inference_confidence",
            y="agent",
            size="player_count",
            color="role_alignment",
            symbol="alignment_stability",
            title="Agent alignment confidence",
            hover_data=["nominal_role", "inferred_behavior_role", "agent_behavior_archetype"],
            color_discrete_map={"aligned": "#52d1b5", "shifted": "#ff6b3d", "insufficient_sample": "#8c95a6"},
        )
        confidence_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
        st.plotly_chart(confidence_fig, use_container_width=True)

    sankey_frame = profiles[["nominal_role", "inferred_behavior_role", "player_count"]].dropna()
    if not sankey_frame.empty:
        links = sankey_frame.groupby(["nominal_role", "inferred_behavior_role"])["player_count"].sum().reset_index()
        labels = pd.Index(pd.concat([links["nominal_role"], links["inferred_behavior_role"]]).unique())
        fig = go.Figure(
            data=[
                go.Sankey(
                    node={"label": labels.tolist(), "color": ["#6bb8ff"] * len(labels)},
                    link={
                        "source": links["nominal_role"].map(labels.get_loc),
                        "target": links["inferred_behavior_role"].map(labels.get_loc),
                        "value": links["player_count"],
                    },
                )
            ]
        )
        fig.update_layout(title="Nominal role to inferred behavior role flow", paper_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
        st.plotly_chart(fig, use_container_width=True)

    selected_agent = st.selectbox("Inspect agent prototype fit", profiles.sort_values("player_count", ascending=False)["agent"].tolist())
    st.plotly_chart(_agent_role_radar(profiles, selected_agent), use_container_width=True)


def _render_segmentation(project_root: Path, results: dict) -> None:
    tab_global, tab_role, tab_agent = st.tabs(["Global", "Role-Specific", "Agent Behavior"])
    with tab_global:
        _global_segmentation_view(project_root, results)
    with tab_role:
        _role_segmentation_view(project_root, results)
    with tab_agent:
        _agent_behavior_view(project_root, results)


def _render_modeling(project_root: Path, results: dict) -> None:
    metrics = _load_json(project_root / "artifacts" / "prediction" / "model_metrics.json")
    calibration = _load_optional_table(project_root / "artifacts" / "prediction" / "calibration_curves.parquet")
    predictions = _load_optional_table(project_root / "artifacts" / "prediction" / "model_predictions.parquet")
    cluster_outcomes = _load_optional_table(project_root / "artifacts" / "segmentation" / "cluster_outcomes.parquet")
    if not metrics:
        st.info("Model artifacts not found.")
        return
    cohort = st.segmented_control("Model cohort", sorted(metrics.keys()), default="pro")
    metric_df = pd.DataFrame([{"model_name": name, **values} for name, values in metrics.get(cohort, {}).items()]).sort_values("brier_score")
    best = results.get("prediction", {}).get("best_models", {}).get(cohort, {})
    _section(
        "Supervised Learnings",
        f"{cohort.title()} outcome modeling",
        f"The strongest current model is <strong>{best.get('model_name', 'n/a')}</strong> with Brier <strong>{_metric(best.get('brier_score'))}</strong>, ROC AUC <strong>{_metric(best.get('roc_auc'))}</strong>, and F1 <strong>{_metric(best.get('f1'))}</strong>. The updated match table now includes team archetype composition features from the unsupervised layer.",
    )
    top_cols = st.columns(min(4, len(metric_df)))
    for idx, row in enumerate(metric_df.head(4).to_dict(orient="records")):
        with top_cols[idx]:
            _card(row["model_name"], _metric(row.get("brier_score")), f"ROC AUC {_metric(row.get('roc_auc'))} | F1 {_metric(row.get('f1'))}")

    chart_cols = [c for c in ["brier_score", "log_loss", "ece", "roc_auc", "average_precision", "accuracy", "f1"] if c in metric_df.columns]
    score_fig = px.bar(metric_df.melt(id_vars="model_name", value_vars=chart_cols, var_name="metric", value_name="value"), x="model_name", y="value", color="metric", barmode="group", title=f"{cohort.title()} model metrics", color_discrete_sequence=px.colors.qualitative.Bold)
    score_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
    st.plotly_chart(score_fig, use_container_width=True)

    left, right = st.columns(2)
    selected_model = st.selectbox("Reliability model", metric_df["model_name"].tolist())
    with left:
        curve = calibration[(calibration["cohort"] == cohort) & (calibration["model_name"] == selected_model)]
        if not curve.empty:
            reliability = px.line(curve, x="mean_predicted", y="empirical_win_rate", markers=True, title=f"{cohort.title()} {selected_model} reliability")
            reliability.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#9aa5b7"}, name="Perfect")
            reliability.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
            st.plotly_chart(reliability, use_container_width=True)
    with right:
        pred_frame = predictions[(predictions["cohort"] == cohort) & (predictions["model_name"] == selected_model)].copy()
        if not pred_frame.empty:
            pred_frame["actual_label"] = pred_frame["actual"].map({1: "Win", 0: "Loss"})
            hist = px.histogram(pred_frame, x="probability", color="actual_label", nbins=18, barmode="overlay", opacity=0.72, title="Predicted probability distribution", color_discrete_map={"Win": "#52d1b5", "Loss": "#ff6b3d"})
            hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
            st.plotly_chart(hist, use_container_width=True)

    if not cluster_outcomes.empty:
        cohort_outcomes = cluster_outcomes[cluster_outcomes["cohort"] == cohort]
        bubble = px.scatter(cohort_outcomes, x="avg_kda_ratio", y="avg_win_rate", size="players", color="archetype_label", title=f"{cohort.title()} cluster outcomes", hover_data=["avg_matches_played"], color_discrete_sequence=px.colors.qualitative.Bold)
        bubble.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
        st.plotly_chart(bubble, use_container_width=True)

    with st.expander("Raw model metrics", expanded=False):
        st.json(metrics)


def _render_comparison(project_root: Path, results: dict) -> None:
    combined_profiles = []
    for cohort in ("pro", "public"):
        frame = _load_optional_table(project_root / "artifacts" / "segmentation" / f"{cohort}_cluster_profiles.parquet")
        if not frame.empty:
            frame["cohort"] = cohort
            combined_profiles.append(frame)
    combined = pd.concat(combined_profiles, ignore_index=True) if combined_profiles else pd.DataFrame()
    cohort_comp = _load_optional_table(project_root / "artifacts" / "segmentation" / "cohort_comparison.parquet")
    if combined.empty:
        st.info("Comparison artifacts not found.")
        return
    best = results.get("prediction", {}).get("best_models", {})
    _section(
        "Supervised vs Unsupervised",
        "What the clusters explain and what the models explain",
        f"Unsupervised clustering gives structure to behavior; supervised modeling tests whether those structures matter for winning. Pro ROC AUC: <strong>{_metric(best.get('pro', {}).get('roc_auc'))}</strong>. Public ROC AUC: <strong>{_metric(best.get('public', {}).get('roc_auc'))}</strong>.",
    )
    left, right = st.columns(2)
    with left:
        _story(
            "Unsupervised finding",
            f"Global silhouette remains moderate, which is expected because behavior overlaps across roles. The role-specific layer is the more faithful VALORANT lens, and the agent-behavior layer checks whether named agents actually behave like their nominal class.",
        )
    with right:
        _story(
            "Supervised finding",
            f"Outcome prediction is much stronger than cluster separation alone. That means match outcomes depend on more than just discrete play-style buckets; continuous player-history signals and team composition matter too.",
        )

    row_a, row_b = st.columns(2)
    with row_a:
        metric = st.selectbox("Cross-cohort trait", [c for c in ["win_rate", "kda_ratio", "support_score", "objective_score", "consistency_score"] if c in combined.columns], index=0)
        fig = px.box(combined, x="cohort", y=metric, color="cohort", points="all", title=f"{metric.replace('_', ' ')} by cohort", color_discrete_map=COHORT_COLORS)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with row_b:
        counts = combined.groupby(["cohort", "archetype_label"])["cluster_size"].sum().reset_index()
        bar = px.bar(counts, x="archetype_label", y="cluster_size", color="cohort", barmode="group", title="Global archetype prevalence", color_discrete_map=COHORT_COLORS)
        bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"}, xaxis_title="")
        st.plotly_chart(bar, use_container_width=True)

    if not cohort_comp.empty:
        delta_fig = px.bar(cohort_comp, x="metric", y="delta_pro_minus_public", title="Pro minus public cluster-metric delta", color="delta_pro_minus_public", color_continuous_scale="RdBu")
        delta_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dbe3f0"})
        st.plotly_chart(delta_fig, use_container_width=True)


def _render_data(project_root: Path) -> None:
    _section("Data Inventory", "Raw, processed, and generated artifacts", "Artifacts stay visible for auditability, but raw structures are collapsed by default.")
    inventory = []
    for label, path in {
        "Raw VCT matches": project_root / "data" / "raw" / "vct" / "vlr_matches.json",
        "Raw Henrik matches": project_root / "data" / "raw" / "public" / "henrik_matches.json",
        "Henrik crawl state": project_root / "data" / "raw" / "public" / "henrik_crawl_state.json",
        "Processed player features": project_root / "data" / "processed" / "player_features.parquet",
        "Processed match level": project_root / "data" / "processed" / "match_level.parquet",
        "Cluster outcomes": project_root / "artifacts" / "segmentation" / "cluster_outcomes.parquet",
        "Prediction metrics": project_root / "artifacts" / "prediction" / "model_metrics.json",
    }.items():
        inventory.append({"artifact": label, "exists": path.exists(), "path": str(path)})
    st.dataframe(pd.DataFrame(inventory), use_container_width=True)
    with st.expander("results.json", expanded=False):
        st.json(_load_json(project_root / "results.json"))


def run_app(project_root: Path) -> None:
    st.set_page_config(page_title="VALORANT Player Intelligence", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(82, 209, 181, 0.12), transparent 18%),
                radial-gradient(circle at top right, rgba(255, 107, 61, 0.12), transparent 22%),
                linear-gradient(180deg, #090b10 0%, #0d1016 50%, #111521 100%);
            color: #e7edf7;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0c0f15 0%, #111520 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        [data-testid="stSidebar"] * { color: #e7edf7; }
        .block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1480px; }
        div[data-testid="stDataFrame"] { background: rgba(16,18,24,0.78); border-radius: 18px; border: 1px solid rgba(255,255,255,0.06); overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    results = _load_json(project_root / "results.json")
    st.markdown("# VALORANT Player Intelligence")
    st.caption("Dark report view for global behavior clusters, role-specific archetypes, agent-behavior alignment, and win modeling on real VALORANT data.")
    page = st.sidebar.radio("Report", REPORT_PAGES)
    if page == "Overview":
        _render_overview(project_root, results)
    elif page == "Segmentation":
        _render_segmentation(project_root, results)
    elif page == "Modeling":
        _render_modeling(project_root, results)
    elif page == "Comparison":
        _render_comparison(project_root, results)
    else:
        _render_data(project_root)


if __name__ == "__main__":
    run_app(Path(__file__).resolve().parents[3])
