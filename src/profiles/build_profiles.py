"""
Build LAD Profile Summary (JTIS v2)

This module is a v2-port of the v1 `lad_profiles.py` logic
(see filecite turn14file0 ).

Behaviour is unchanged, but the implementation is modernised:

    - Accepts DataFrames (no file paths)
    - No writing to CSV (pipeline handles this)
    - Clean imports and modularisation
    - Clustering moved to `cluster_labels.py`

Returned DataFrame contains:
    - trend metrics
    - national percentiles
    - component percentiles
    - ranks
    - priority groups
    - cluster_id + cluster_label
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.profiles.cluster_labels import (
    kmeans_cluster_profiles,
    PROFILE_CLUSTER_FEATURES,
    map_cluster_label,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Utility functions ported directly from v1
# ---------------------------------------------------------------------

_LAD_NAME_CANDIDATES = [
    "lad_name",
    "LAD_NAME",
    "lad_name_x",
    "lad_name_y",
]


def _detect_lad_name_column(df: pd.DataFrame):
    for col in _LAD_NAME_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _classify_trend(delta: float, threshold: float = 0.05) -> str:
    if pd.isna(delta):
        return "Unknown"
    if delta > threshold:
        return "Improving"
    if delta < -threshold:
        return "Worsening"
    return "Stable"


def _percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100


# ---------------------------------------------------------------------
# Main v2 profiling function
# ---------------------------------------------------------------------

def build_lad_profiles(jti_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build LAD profile summary from the full JTI dataset.

    Equivalent to v1 `build_lad_profile_summary` but takes a DataFrame.

    Parameters
    ----------
    jti_df : DataFrame
        Must contain:
            lad_code, year, jti_score,
            component_emissions_last,
            component_transport_last,
            component_deprivation_last

    Returns
    -------
    DataFrame
        LAD-level summary table.
    """
    df = jti_df.copy()
    logger.info("Building LAD profiles from in-memory JTI df (%d rows)", len(df))

    # Basic validation
    required = {"lad_code", "year", "jti_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    lad_name_col = _detect_lad_name_column(df)

    summary_records = []

    for lad_code, grp in df.groupby("lad_code"):
        grp = grp.sort_values("year")
        if grp["year"].isna().all():
            continue

        first_year = int(grp["year"].min())
        last_year = int(grp["year"].max())
        n_years = grp["year"].nunique()

        first_row = grp.loc[grp["year"] == first_year].iloc[0]
        last_row = grp.loc[grp["year"] == last_year].iloc[0]

        jti_first = float(first_row["jti_score"])
        jti_last = float(last_row["jti_score"])
        jti_delta = jti_last - jti_first

        summary_records.append(
            {
                "lad_code": lad_code,
                "lad_name": str(last_row.get(lad_name_col)) if lad_name_col else None,
                "first_year": first_year,
                "last_year": last_year,
                "n_years": n_years,
                "jti_first": jti_first,
                "jti_last": jti_last,
                "jti_delta": jti_delta,
                "trend_label": _classify_trend(jti_delta),
                "component_emissions_last": float(last_row.get("component_emissions_last", np.nan)),
                "component_transport_last": float(last_row.get("component_transport_last", np.nan)),
                "component_deprivation_last": float(last_row.get("component_deprivation_last", np.nan)),
            }
        )

    summary_df = pd.DataFrame(summary_records)

    # Percentiles (v1)
    summary_df["jti_last_percentile"] = _percentile_rank(summary_df["jti_last"])
    summary_df["component_emissions_percentile"] = _percentile_rank(
        summary_df["component_emissions_last"]
    )
    summary_df["component_transport_percentile"] = _percentile_rank(
        summary_df["component_transport_last"]
    )
    summary_df["component_deprivation_percentile"] = _percentile_rank(
        summary_df["component_deprivation_last"]
    )

    # National ranks (1 = highest score)
    summary_df["jti_last_rank"] = summary_df["jti_last"].rank(
        ascending=False
    ).astype("Int64")
    summary_df["component_emissions_rank"] = summary_df["component_emissions_last"].rank(
        ascending=False
    ).astype("Int64")
    summary_df["component_transport_rank"] = summary_df["component_transport_last"].rank(
        ascending=False
    ).astype("Int64")
    summary_df["component_deprivation_rank"] = summary_df["component_deprivation_last"].rank(
        ascending=False
    ).astype("Int64")

    # Priority grouping (v1 logic)
    def classify_priority(row):
        if (
            row["trend_label"] == "Worsening"
            or row["component_deprivation_percentile"] >= 70
            or row["jti_last_percentile"] <= 30
        ):
            return "High Priority"

        if (
            row["trend_label"] == "Improving"
            and row["jti_last_percentile"] >= 70
            and row["component_deprivation_percentile"] <= 30
        ):
            return "Low Priority"

        return "Medium Priority"

    summary_df["priority_group"] = summary_df.apply(classify_priority, axis=1)

    # Cluster IDs + labels
    cluster_ids, mask = kmeans_cluster_profiles(summary_df)
    summary_df["cluster_id"] = np.nan
    summary_df.loc[mask, "cluster_id"] = cluster_ids
    summary_df["cluster_label"] = summary_df["cluster_id"].apply(map_cluster_label)

    logger.info("Profile building complete (%d LADs)", len(summary_df))
    return summary_df
