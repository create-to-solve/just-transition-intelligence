"""
JTIS v2 Dashboard (Streamlit)

Executive Summary + Components page
Neutral grey/white modern data portal style

Tabs:
 - Overview (Executive Summary)
 - Components (component KPI + distributions + radar)
 - Trends (placeholder)
 - Comparisons (placeholder)
 - Clusters (placeholder)

Reads from:
 - outputs/jti_scores.csv
 - outputs/lad_profile_summary.csv

Run:
  streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import os
import altair as alt
import pandas as pd
import streamlit as st
import numpy as np

# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(
    page_title="JTIS Dashboard",
    layout="wide",
    page_icon="📊",
)

# --------------------------------------------------
# Global CSS – neutral, modern cards
# --------------------------------------------------

st.markdown(
    """
<style>
:root {
  --card-bg: #ffffff;
  --card-border: #e5e5e5;
  --card-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
  --pill-bg: #f5f5f5;
  --pill-text: #333333;
  --accent-good: #0f766e;
  --accent-neutral: #4b5563;
  --accent-bad: #b91c1c;
}

/* Metric card container */
div.metric-card {
  background: var(--card-bg);
  border-radius: 16px;
  padding: 1rem 1.25rem;
  border: 1px solid var(--card-border);
  box-shadow: var(--card-shadow);
}

/* Metric heading */
div.metric-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #6b7280;
  margin-bottom: 0.25rem;
}

/* Main value */
div.metric-value {
  font-size: 1.7rem;
  font-weight: 600;
  color: #111827;
}

/* Subtext */
div.metric-sub {
  font-size: 0.9rem;
  color: #6b7280;
  margin-top: 0.25rem;
}

/* Pill tags */
span.pill {
  display: inline-block;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  background: var(--pill-bg);
  color: var(--pill-text);
  font-size: 0.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Data loading
# --------------------------------------------------

@st.cache_data
def load_jti_scores(path: str = "outputs/jti_scores.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_profiles(path: str = "outputs/lad_profile_summary.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

# --------------------------------------------------
# Small chart helpers
# --------------------------------------------------

def jti_trend_chart(jti_lad: pd.DataFrame) -> alt.Chart:
    """Simple JTI score over time chart for a single LAD."""
    if jti_lad.empty:
        return alt.Chart(pd.DataFrame({"year": [], "jti_score": []})).mark_line()

    return (
        alt.Chart(jti_lad)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("jti_score:Q", title="JTI score"),
            tooltip=["year", "jti_score"],
        )
        .properties(height=260, title="JTI score over time")
    )

# --------------------------------------------------
# Executive Summary renderer
# --------------------------------------------------

def render_executive_summary(
    jti_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    selected_lad: str,
) -> None:

    profile_rows = profiles_df[profiles_df["lad_name"] == selected_lad]
    if profile_rows.empty:
        st.warning("No profile row found for this LAD.")
        return

    lad_profile = profile_rows.iloc[0]

    total_lads = profiles_df["lad_code"].nunique() if "lad_code" in profiles_df else len(profiles_df)

    jti_last = lad_profile.get("jti_last", float("nan"))
    jti_rank = lad_profile.get("jti_last_rank", float("nan"))
    jti_pct = lad_profile.get("jti_last_percentile", float("nan"))

    comp_emis = lad_profile.get("component_emissions_last", float("nan"))
    comp_tran = lad_profile.get("component_transport_last", float("nan"))
    comp_depr = lad_profile.get("component_deprivation_last", float("nan"))

    comp_emis_pct = lad_profile.get("component_emissions_percentile", float("nan"))
    comp_tran_pct = lad_profile.get("component_transport_percentile", float("nan"))
    comp_depr_pct = lad_profile.get("component_deprivation_percentile", float("nan"))

    priority = lad_profile.get("priority_group", "Unknown")
    cluster = lad_profile.get("cluster_label", "Not assigned")
    trend = lad_profile.get("trend_label", "Stable")

    jti_lad = jti_df[jti_df.get("lad_name", "") == selected_lad]
    if jti_lad.empty and "lad_code" in lad_profile and "lad_code" in jti_df:
        jti_lad = jti_df[jti_df["lad_code"] == lad_profile["lad_code"]]

    last_year = int(jti_lad["year"].max()) if not jti_lad.empty else None

    # Safe formatting
    jti_last_f = f"{jti_last:.1f}" if pd.notna(jti_last) else "–"
    comp_emis_f = f"{comp_emis:.1f}" if pd.notna(comp_emis) else "–"
    comp_tran_f = f"{comp_tran:.1f}" if pd.notna(comp_tran) else "–"
    comp_depr_f = f"{comp_depr:.1f}" if pd.notna(comp_depr) else "–"

    st.markdown(
        f"### Executive summary — {selected_lad}"
        + (f" (latest year: {last_year})" if last_year else "")
    )

    st.markdown(
        "This page provides a high-level view of how this local authority is "
        "positioned within the UK Just Transition Intelligence System."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    # ---------------- JTI Score ----------------
    with col1:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">JTI score</div>
  <div class="metric-value">{jti_last_f}</div>
  <div class="metric-sub">Latest composite just transition score.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ---------------- Rank ----------------
    with col2:
        rank_text = (
            f"{int(jti_rank)} of {int(total_lads)}"
            if pd.notna(jti_rank) and pd.notna(total_lads)
            else "–"
        )
        pct_text = (
            f"Approx. {int(jti_pct)}th percentile"
            if pd.notna(jti_pct)
            else "Percentile not available"
        )

        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">National position</div>
  <div class="metric-value">{rank_text}</div>
  <div class="metric-sub">{pct_text}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ---------------- Priority ----------------
    with col3:
        if isinstance(priority, str):
            if "High" in priority:
                priority_color = "var(--accent-bad)"
            elif "Low" in priority:
                priority_color = "var(--accent-good)"
            else:
                priority_color = "var(--accent-neutral)"
        else:
            priority_color = "var(--accent-neutral)"

        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Priority & cluster</div>
  <div class="metric-value" style="font-size: 1.1rem;">
    <span class="pill" style="border: 1px solid {priority_color}; color: {priority_color};">
      {priority}
    </span>
  </div>
  <div class="metric-sub">
    Cluster: <strong>{cluster}</strong><br/>
    Overall trend: <strong>{trend}</strong>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ---------------- Component Cards ----------------
    st.markdown("### Component picture")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Emissions</div>
  <div class="metric-value">{comp_emis_f}</div>
  <div class="metric-sub">
    Percentile: {int(comp_emis_pct) if pd.notna(comp_emis_pct) else "–"}th
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Transport</div>
  <div class="metric-value">{comp_tran_f}</div>
  <div class="metric-sub">
    Percentile: {int(comp_tran_pct) if pd.notna(comp_tran_pct) else "–"}th
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Deprivation</div>
  <div class="metric-value">{comp_depr_f}</div>
  <div class="metric-sub">
    Percentile: {int(comp_depr_pct) if pd.notna(comp_depr_pct) else "–"}th
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ---------------- Chart + Narrative ----------------
    chart_col, text_col = st.columns((2, 1))

    with chart_col:
        st.markdown("### JTI trajectory")
        st.altair_chart(jti_trend_chart(jti_lad), use_container_width=True)

    with text_col:
        st.markdown("### What this means")

        narrative = []

        if pd.notna(jti_pct):
            if jti_pct >= 70:
                narrative.append(
                    "This area sits in the **upper band nationally** for overall JTI."
                )
            elif jti_pct <= 30:
                narrative.append(
                    "This area sits in the **lower band nationally** for overall JTI."
                )
            else:
                narrative.append(
                    "This area sits around the **middle of the national distribution**."
                )

        if isinstance(trend, str):
            if "Improving" in trend:
                narrative.append(
                    "Recent years show an **improving trend** in conditions for a just transition."
                )
            elif "Worsening" in trend:
                narrative.append(
                    "Recent years show a **worsening trend**, suggesting emerging challenges."
                )

        if isinstance(priority, str):
            if "High" in priority:
                narrative.append(
                    "This area is flagged as **High Priority** for just transition support."
                )
            elif "Low" in priority:
                narrative.append(
                    "This area is currently classified as **Low Priority**, but monitoring remains important."
                )

        if not narrative:
            narrative.append("No automated narrative available.")

        st.markdown("\n\n".join(f"- {line}" for line in narrative))


# --------------------------------------------------
# Components page
# --------------------------------------------------

def render_components_page(profiles_df, selected_lad):

    st.subheader("Component Breakdown")

    profile_rows = profiles_df[profiles_df["lad_name"] == selected_lad]
    if profile_rows.empty:
        st.warning("No profile row found.")
        return

    lad_profile = profile_rows.iloc[0]

    emis = lad_profile.get("component_emissions_last", float("nan"))
    tran = lad_profile.get("component_transport_last", float("nan"))
    depr = lad_profile.get("component_deprivation_last", float("nan"))

    emis_pct = lad_profile.get("component_emissions_percentile", float("nan"))
    tran_pct = lad_profile.get("component_transport_percentile", float("nan"))
    depr_pct = lad_profile.get("component_deprivation_percentile", float("nan"))

    emis_f = f"{emis:.1f}" if pd.notna(emis) else "–"
    tran_f = f"{tran:.1f}" if pd.notna(tran) else "–"
    depr_f = f"{depr:.1f}" if pd.notna(depr) else "–"

    # ---------------- KPI cards ----------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Emissions</div>
  <div class="metric-value">{emis_f}</div>
  <div class="metric-sub">
    Percentile: {int(emis_pct) if pd.notna(emis_pct) else "–"}th
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Transport</div>
  <div class="metric-value">{tran_f}</div>
  <div class="metric-sub">
    Percentile: {int(tran_pct) if pd.notna(tran_pct) else "–"}th
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Deprivation</div>
  <div class="metric-value">{depr_f}</div>
  <div class="metric-sub">
    Percentile: {int(depr_pct) if pd.notna(depr_pct) else "–"}th
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Component Distributions")

    # ------------------------------------------
    # National distribution charts
    # ------------------------------------------

    def comp_hist(df, col, lad_val, label):
        base = alt.Chart(df).transform_filter(
            alt.datum[col] != None
        )

        hist = (
            base.mark_bar(color="#d1d5db")
            .encode(
                x=alt.X(f"{col}:Q", bin=True, title=f"{label} score"),
                y=alt.Y("count()", title="Number of LADs"),
            )
            .properties(height=180)
        )

        marker = (
            alt.Chart(pd.DataFrame({col: [lad_val]}))
            .mark_rule(color="#111827", strokeWidth=2)
            .encode(x=alt.X(f"{col}:Q"))
        )

        return hist + marker

    h1, h2, h3 = st.columns(3)

    with h1:
        st.markdown("**Emissions distribution**")
        st.altair_chart(
            comp_hist(profiles_df, "component_emissions_last", emis, "Emissions"),
            use_container_width=True,
        )

    with h2:
        st.markdown("**Transport distribution**")
        st.altair_chart(
            comp_hist(profiles_df, "component_transport_last", tran, "Transport"),
            use_container_width=True,
        )

    with h3:
        st.markdown("**Deprivation distribution**")
        st.altair_chart(
            comp_hist(profiles_df, "component_deprivation_last", depr, "Deprivation"),
            use_container_width=True,
        )

    # ------------------------------------------
    # Radar chart
    # ------------------------------------------


    # ------------------------------------------
    # Narrative
    # ------------------------------------------

    st.markdown("### Interpretation")

    comp_dict = {
        "Emissions": emis_pct,
        "Transport": tran_pct,
        "Deprivation": depr_pct,
    }

    strengths = [k for k, v in comp_dict.items() if pd.notna(v) and v >= 66]
    weaknesses = [k for k, v in comp_dict.items() if pd.notna(v) and v <= 33]

    narrative = []

    if strengths:
        narrative.append(
            f"**Strengths:** {', '.join(strengths)} are comparatively strong components."
        )
    if weaknesses:
        narrative.append(
            f"**Challenges:** {', '.join(weaknesses)} appear weaker and may require targeted support."
        )
    if not narrative:
        narrative.append("Component performance is broadly mid-range across the board.")

    st.markdown("\n\n".join(f"- {line}" for line in narrative))


# --------------------------------------------------
# Main app
# --------------------------------------------------

def main() -> None:
    st.title("📊 Just Transition Intelligence System — Dashboard (v2)")

    jti_df = load_jti_scores()
    profiles_df = load_profiles()

    if jti_df.empty or profiles_df.empty:
        st.warning("Missing required output files. Please run the pipeline first.")
        return

    st.sidebar.header("Filters")

    lad_list = sorted(profiles_df["lad_name"].dropna().unique())
    selected_lad = st.sidebar.selectbox("Select Local Authority", lad_list)

    tab_overview, tab_components, tab_trends, tab_compare, tab_clusters = st.tabs(
        ["Overview", "Components", "Trends", "Comparisons", "Clusters"]
    )

    with tab_overview:
        render_executive_summary(jti_df, profiles_df, selected_lad)

    with tab_components:
        render_components_page(profiles_df, selected_lad)




    with tab_trends:
        st.subheader("Trends over time")

        # Filter long-form JTI scores for this LAD
        lad_ts = jti_df[jti_df["lad_name"] == selected_lad].sort_values("year")

        if lad_ts.empty:
            st.warning("No time-series data available for this LAD.")
            st.stop()

        # Column availability
        has_emis = "per_capita_emissions" in lad_ts
        has_tran = "transport_mwh_per_capita" in lad_ts
        has_imd = "imd_rank_mean" in lad_ts

        # --- % CHANGE helpers ---
        def pct_change(df, col):
            try:
                first = df[col].iloc[0]
                last = df[col].iloc[-1]
                if pd.isna(first) or first == 0:
                    return float("nan")
                return ((last - first) / first) * 100
            except Exception:
                return float("nan")

        emis_change = pct_change(lad_ts, "per_capita_emissions") if has_emis else float("nan")
        tran_change = pct_change(lad_ts, "transport_mwh_per_capita") if has_tran else float("nan")
        imd_change  = pct_change(lad_ts, "imd_rank_mean") if has_imd else float("nan")

        # --- KPI CARDS ---
        st.markdown("### Change since first year")

        c1, c2, c3 = st.columns(3)

        with c1:
            emis_f = f"{emis_change:.1f}%" if pd.notna(emis_change) else "–"
            st.markdown(
                f"""
    <div class="metric-card">
    <div class="metric-label">Per-capita emissions</div>
    <div class="metric-value">{emis_f}</div>
    <div class="metric-sub">Change from first to latest year</div>
    </div>
    """,
                unsafe_allow_html=True,
            )

        with c2:
            tran_f = f"{tran_change:.1f}%" if pd.notna(tran_change) else "–"
            st.markdown(
                f"""
    <div class="metric-card">
    <div class="metric-label">Transport (MWh per capita)</div>
    <div class="metric-value">{tran_f}</div>
    <div class="metric-sub">Change from first to latest year</div>
    </div>
    """,
                unsafe_allow_html=True,
            )

        with c3:
            imd_f = f"{imd_change:.1f}%" if pd.notna(imd_change) else "–"
            st.markdown(
                f"""
    <div class="metric-card">
    <div class="metric-label">IMD (mean rank)</div>
    <div class="metric-value">{imd_f}</div>
    <div class="metric-sub">Change from first to latest year</div>
    </div>
    """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # --- Charts ---

        if has_emis:
            st.markdown("### Per-capita emissions over time")
            chart_emis = (
                alt.Chart(lad_ts)
                .mark_line(point=True)
                .encode(
                    x="year:O",
                    y=alt.Y("per_capita_emissions:Q", title="tCO₂e per person"),
                    tooltip=["year", "per_capita_emissions"]
                )
                .properties(height=260)
            )
            st.altair_chart(chart_emis, use_container_width=True)
        else:
            st.info("No emissions time-series for this LAD.")

        if has_tran:
            st.markdown("### Transport energy use (MWh per capita) over time")
            chart_tran = (
                alt.Chart(lad_ts)
                .mark_line(point=True)
                .encode(
                    x="year:O",
                    y=alt.Y("transport_mwh_per_capita:Q", title="MWh per person"),
                    tooltip=["year", "transport_mwh_per_capita"]
                )
                .properties(height=260)
            )
            st.altair_chart(chart_tran, use_container_width=True)
        else:
            st.info("No transport time-series for this LAD.")

        if has_imd:
            st.markdown("### Deprivation (IMD mean rank) over time")
            chart_imd = (
                alt.Chart(lad_ts)
                .mark_line(point=True)
                .encode(
                    x="year:O",
                    y=alt.Y("imd_rank_mean:Q", title="IMD rank (mean)"),
                    tooltip=["year", "imd_rank_mean"]
                )
                .properties(height=260)
            )
            st.altair_chart(chart_imd, use_container_width=True)
        else:
            st.info("No deprivation time-series for this LAD.")




    with tab_compare:
        st.subheader("Peer Comparisons & Benchmarking")

        # Fetch LAD profile row
        lad_profile = profiles_df[profiles_df["lad_name"] == selected_lad].iloc[0]


        # ================
        # 1. NATIONAL POSITION
        # ================

        st.markdown("### National distribution (latest year)")

        latest_year = jti_df["year"].max()
        national_latest = jti_df[jti_df["year"] == latest_year]

        if national_latest.empty:
            st.warning("No national distribution data available.")
            st.stop()

        # Sort by JTI score
        national_sorted = national_latest.sort_values("jti_score")

        # Percentile for the selected LAD
        lad_row = national_sorted[national_sorted["lad_name"] == selected_lad]
        if lad_row.empty:
            st.warning("Selected LAD not found in latest-year national distribution.")
            st.stop()

        lad_percentile = (
            lad_profile["jti_last_percentile"]
            if "jti_last_percentile" in lad_profile.index
            else None
        )

        # Highlight selected LAD
        national_sorted["highlight"] = national_sorted["lad_name"] == selected_lad

        # Altair chart
        rank_chart = (
            alt.Chart(national_sorted)
            .mark_bar()
            .encode(
                y=alt.Y("lad_name:N", sort="-x", title="Local Authority"),
                x=alt.X("jti_score:Q", title="JTI score"),
                color=alt.condition(
                    alt.datum.highlight,
                    alt.value("#111827"),     # selected LAD
                    alt.value("#d1d5db")      # others
                ),
                tooltip=["lad_name", "jti_score"]
            )
            .properties(height=800)
        )

        st.altair_chart(rank_chart, use_container_width=True)

        # Percentile card
        perc_val = (
            f"{int(lad_percentile)}th" if lad_percentile is not None else "–"
        )

        st.markdown(
            f"""
    <div class="metric-card" style="margin-top:1rem; max-width:260px;">
    <div class="metric-label">National percentile</div>
    <div class="metric-value">{perc_val}</div>
    <div class="metric-sub">Position within all LADs ({latest_year})</div>
    </div>
    """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ================
        # 2. CLUSTER PEERS
        # ================

        st.markdown("### Cluster peers")

        cluster_id = lad_profile["cluster_label"]

        cluster_df = profiles_df[profiles_df["cluster_label"] == cluster_id]

        st.markdown(f"**Cluster: {cluster_id}** — {len(cluster_df)} authorities")

        # Bar chart vs cluster peers
        cluster_latest = national_latest[
            national_latest["lad_name"].isin(cluster_df["lad_name"])
        ].sort_values("jti_score")

        if not cluster_latest.empty:
            cluster_chart = (
                alt.Chart(cluster_latest)
                .mark_bar()
                .encode(
                    x=alt.X("jti_score:Q", title="JTI score"),
                    y=alt.Y("lad_name:N", sort="-x"),
                    color=alt.condition(
                        alt.datum.lad_name == selected_lad,
                        alt.value("#111827"),
                        alt.value("#d1d5db")
                    ),
                    tooltip=["lad_name", "jti_score"],
                )
                .properties(height=max(200, 30 * len(cluster_latest)))
            )
            st.altair_chart(cluster_chart, use_container_width=True)
        else:
            st.info("No cluster-matched data for this LAD.")

        st.markdown("---")

        # ================
        # 3. STATISTICAL PEERS (nearest LADs)
        # ================

        st.markdown("### Statistical peers")

        # We use 3 dimensions for closeness:
        # - Latest JTI score
        # - Latest emission component
        # - Latest transport component

        def get_distance(row):
            return (
                (row["component_emissions_last"] - lad_profile["component_emissions_last"]) ** 2
                + (row["component_transport_last"] - lad_profile["component_transport_last"]) ** 2
                + (row["component_deprivation_last"] - lad_profile["component_deprivation_last"]) ** 2
            ) ** 0.5

        tmp = profiles_df.copy()
        tmp["distance"] = tmp.apply(get_distance, axis=1)

        peers = (
            tmp[tmp["lad_name"] != selected_lad]
            .sort_values("distance")
            .head(5)
        )

        st.markdown("Closest **5 statistical peers** based on components:")

        st.dataframe(
            peers[["lad_name", "component_emissions_last", "component_transport_last", "component_deprivation_last"]],
            use_container_width=True
        )

        st.markdown("---")

        # ================
        # 4. MINI CHART: LAD vs PEER RANGE
        # ================

        st.markdown("### LAD vs statistical peers — component comparison")

        # Build tidy DF
        component_names = ["component_emissions_last", "component_transport_last", "component_deprivation_last"]
        pretty_names = ["Emissions", "Transport", "Deprivation"]

        comp_df = pd.DataFrame({
            "component": pretty_names,
            "LAD": [
                lad_profile["component_emissions_last"],
                lad_profile["component_transport_last"],
                lad_profile["component_deprivation_last"],
            ],
            "Peer_min": [
                peers["component_emissions_last"].min(),
                peers["component_transport_last"].min(),
                peers["component_deprivation_last"].min(),
            ],
            "Peer_max": [
                peers["component_emissions_last"].max(),
                peers["component_transport_last"].max(),
                peers["component_deprivation_last"].max(),
            ],
            "Peer_mean": [
                peers["component_emissions_last"].mean(),
                peers["component_transport_last"].mean(),
                peers["component_deprivation_last"].mean(),
            ],
        })

        # Melt for charting
        melt_df = comp_df.melt(id_vars="component", var_name="type", value_name="value")

        # Plot: range (min/max) + mean + LAD marker
        range_chart = (
            alt.Chart(melt_df)
            .mark_line()
            .encode(
                x=alt.X("value:Q"),
                y=alt.Y("component:N"),
                color=alt.Color("type:N", scale=alt.Scale(range=["#d1d5db", "#9ca3af", "#111827"]))
            )
        )

        st.altair_chart(range_chart, use_container_width=True)


    with tab_clusters:
        st.subheader("Cluster Profile")

        # ----------------------------
        # Fetch LAD profile
        # ----------------------------
        lad_profile = profiles_df[profiles_df["lad_name"] == selected_lad].iloc[0]
        lad_cluster = lad_profile["cluster_label"]

        st.markdown(f"### Cluster: **{lad_cluster}**")

        # Subset of LADs in this cluster
        cluster_members = profiles_df[profiles_df["cluster_label"] == lad_cluster]

        if cluster_members.empty:
            st.warning("No cluster members found for this LAD.")
            st.stop()

        # ----------------------------
        # Top Card — Cluster Size
        # ----------------------------
        st.markdown(
            f"""
    <div class="metric-card" style="max-width: 300px; margin-bottom: 1rem;">
    <div class="metric-label">Cluster size</div>
    <div class="metric-value">{len(cluster_members)}</div>
    <div class="metric-sub">Number of local authorities in this cluster</div>
    </div>
    """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ----------------------------
        # SUMMARY STATISTICS
        # ----------------------------

        st.markdown("### Cluster characteristics")

        summary_df = cluster_members[
            [
                "component_emissions_last",
                "component_transport_last",
                "component_deprivation_last",
                "jti_last",
            ]
        ].agg(["mean", "min", "max"]).T
        summary_df.columns = ["Mean", "Min", "Max"]

        st.dataframe(summary_df, use_container_width=True)

        st.markdown("---")

        # ----------------------------
        # LAD vs Cluster Benchmarks — Component Bars
        # ----------------------------

        st.markdown("### Your area vs cluster peers")

        comp_names = {
            "component_emissions_last": "Emissions",
            "component_transport_last": "Transport",
            "component_deprivation_last": "Deprivation",
            "jti_last": "JTI score",
        }

        bench_df = pd.DataFrame({
            "component": list(comp_names.values()),
            "LAD": [
                lad_profile["component_emissions_last"],
                lad_profile["component_transport_last"],
                lad_profile["component_deprivation_last"],
                lad_profile["jti_last"],
            ],
            "Cluster_mean": [
                cluster_members["component_emissions_last"].mean(),
                cluster_members["component_transport_last"].mean(),
                cluster_members["component_deprivation_last"].mean(),
                cluster_members["jti_last"].mean(),
            ],
            "Cluster_min": [
                cluster_members["component_emissions_last"].min(),
                cluster_members["component_transport_last"].min(),
                cluster_members["component_deprivation_last"].min(),
                cluster_members["jti_last"].min(),
            ],
            "Cluster_max": [
                cluster_members["component_emissions_last"].max(),
                cluster_members["component_transport_last"].max(),
                cluster_members["component_deprivation_last"].max(),
                cluster_members["jti_last"].max(),
            ],
        })

        # Tidy format
        bench_melt = bench_df.melt(id_vars="component", var_name="group", value_name="value")

        comp_chart = (
            alt.Chart(bench_melt)
            .mark_point(filled=True, size=80)
            .encode(
                y=alt.Y("component:N", title=""),
                x=alt.X("value:Q", title="Score"),
                color=alt.Color(
                    "group:N",
                    scale=alt.Scale(
                        domain=["LAD", "Cluster_mean", "Cluster_min", "Cluster_max"],
                        range=["#111827", "#4b5563", "#d1d5db", "#d1d5db"],
                    ),
                    legend=alt.Legend(title="")
                ),
                tooltip=["component", "group", "value"],
            )
            .properties(height=200)
        )

        st.altair_chart(comp_chart, use_container_width=True)

        st.markdown("---")

        # ----------------------------
        # DISTRIBUTIONS WITH HIGHLIGHT
        # ----------------------------

        st.markdown("### Cluster distributions")

        for col, label in comp_names.items():
            st.markdown(f"#### {label}")

            df = cluster_members[[col, "lad_name"]].dropna()

            # Add highlight flag
            df["highlight"] = df["lad_name"] == selected_lad

            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=20), title=f"{label} score"),
                    y=alt.Y("count()", title="Count"),
                    color=alt.condition("datum.highlight", alt.value("#111827"), alt.value("#d1d5db")),
                    tooltip=["lad_name", col]
                )
                .properties(height=200)
            )

            st.altair_chart(chart, use_container_width=True)

        st.markdown("---")

        # ----------------------------
        # NARRATIVE
        # ----------------------------

        st.markdown("### Interpretation")

        narrative_lines = []

        # Compare LAD to cluster mean
        for col, label in comp_names.items():
            lad_val = lad_profile[col]
            cluster_mean = cluster_members[col].mean()

            if pd.notna(lad_val):
                if lad_val > cluster_mean:
                    narrative_lines.append(f"**{label}**: Your area is **above** the cluster average.")
                elif lad_val < cluster_mean:
                    narrative_lines.append(f"**{label}**: Your area is **below** the cluster average.")
                else:
                    narrative_lines.append(f"**{label}**: Your area is close to the cluster norm.")

        if not narrative_lines:
            narrative_lines.append("No narrative could be generated for this cluster.")

        st.markdown("\n\n".join(f"- {line}" for line in narrative_lines))


if __name__ == "__main__":
    main()
