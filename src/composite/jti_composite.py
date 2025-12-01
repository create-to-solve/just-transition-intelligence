"""
JTIS v2 Composite Score Module

This file contains:
    1. The original v1 JTI component scaling + scoring logic
       (ported exactly from the v1 repository with no changes).
    2. A v2 wrapper (`build_jti_scores`) which:
        - assembles component indicators
        - applies the v1 scoring logic
        - returns a final LAD–year JTI dataset

No core logic has been changed — only structured and modernised.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.indicators.composite_indicators import assemble_composite_inputs
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# v1 Logic (ported exactly)
# ---------------------------------------------------------------------

def min_max_scale(series: pd.Series) -> pd.Series:
    """
    Min–max scale a pandas Series between 0 and 100.
    Identical to v1 behaviour.
    """
    scaler = MinMaxScaler(feature_range=(0, 100))
    reshaped = series.values.reshape(-1, 1)
    scaled = scaler.fit_transform(reshaped)
    return pd.Series(scaled.flatten(), index=series.index)


def compute_jti_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute scaled components and final JTI score.
    This is the v1 composite scoring algorithm.

    Expected columns:
        - per_capita_emissions
        - transport_mwh_per_capita
        - imd_rank_mean
    """
    required = {
        "lad_code",
        "year",
        "per_capita_emissions",
        "transport_mwh_per_capita",
        "imd_rank_mean",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_jti_scores: missing required columns: {missing}"
        )

    out = df.copy()

    # Scale components (0–100)
    out["per_capita_emissions_scaled"] = min_max_scale(
        out["per_capita_emissions"]
    )
    out["transport_scaled"] = min_max_scale(
        out["transport_mwh_per_capita"]
    )
    out["deprivation_scaled"] = min_max_scale(
        out["imd_rank_mean"]
    )

    # Final JTI score: simple mean of scaled components (v1 logic)
    out["jti_score"] = (
        out[
            [
                "per_capita_emissions_scaled",
                "transport_scaled",
                "deprivation_scaled",
            ]
        ].mean(axis=1)
    )

    return out


# ---------------------------------------------------------------------
# v2 Wrapper (new)
# ---------------------------------------------------------------------

def build_jti_scores(
    ghg_df: pd.DataFrame,
    transport_df: pd.DataFrame,
    deprivation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full JTIS v2 workflow:
        1. Assemble component-level LAD–year indicators.
        2. Apply v1 JTI scoring logic.

    Parameters
    ----------
    ghg_df : harmonised emissions + per-capita indicators
    transport_df : harmonised transport + per-capita indicators
    deprivation_df : LAD-level IMD indicators

    Returns
    -------
    DataFrame with LAD–year JTI score and scaled components.
    """
    logger.info("Building JTI scores (v2)...")

    # Step 1: Build component inputs
    components = assemble_composite_inputs(
        ghg_df=ghg_df,
        transport_df=transport_df,
        deprivation_df=deprivation_df,
    )

    # Step 2: Compute scaled components + final JTI score
    scored = compute_jti_scores(components)

    logger.info("JTI scoring complete: %d rows", len(scored))
    return scored
