"""
Deprivation indicators (v2 port, identical behaviour to v1).

Includes:
    - IMD LSOA → LAD aggregation
    - Rank-based deprivation measure
    - LAD-level IMD percentiles

Expected input:
    imd_lsoa (raw IMD2019 rows)

Outputs:
    - LAD-level IMD deprivation scores
    - LAD-level IMD percentiles
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger
from src.utils.percentile import add_percentiles

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# 1. IMD aggregation: LSOA → LAD
# ---------------------------------------------------------------------

def imd_aggregate(imd_lsoa: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate IMD 2019 from LSOA-level to LAD-level.

    Behaviour exactly matches v1:
        - For each LAD, compute mean IMD rank and mean IMD decile.
        - Produces LAD-only (no year dimension).
    """
    required = {"lad_code", "imd_rank", "imd_decile"}
    missing = required - set(imd_lsoa.columns)
    if missing:
        logger.warning("imd_aggregate: missing columns: %s", missing)
        return pd.DataFrame(columns=["lad_code", "imd_rank_mean", "imd_decile_mean"])

    df = imd_lsoa.copy()
    df["imd_rank"] = pd.to_numeric(df["imd_rank"], errors="coerce")
    df["imd_decile"] = pd.to_numeric(df["imd_decile"], errors="coerce")

    agg = (
        df.groupby("lad_code")[["imd_rank", "imd_decile"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "imd_rank": "imd_rank_mean",
                "imd_decile": "imd_decile_mean",
            }
        )
    )

    logger.info("IMD aggregate: %d LADs", len(agg))
    return agg


# ---------------------------------------------------------------------
# 2. Deprivation score (rank-based)
# ---------------------------------------------------------------------

def deprivation_score(imd_lad: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a rank-based deprivation score from mean IMD rank.

    Lower IMD rank → more deprived.
    So we invert and scale to 0–1:
        deprivation_score = 1 - (rank_mean / max_rank)
    """
    if "imd_rank_mean" not in imd_lad.columns:
        logger.warning("deprivation_score: missing imd_rank_mean")
        imd_lad = imd_lad.copy()
        imd_lad["deprivation_score"] = None
        return imd_lad

    df = imd_lad.copy()
    max_rank = df["imd_rank_mean"].max()
    df["deprivation_score"] = 1 - (df["imd_rank_mean"] / max_rank)
    return df


# ---------------------------------------------------------------------
# 3. IMD percentiles (0–100)
# ---------------------------------------------------------------------

def deprivation_percentiles(imd_lad: pd.DataFrame) -> pd.DataFrame:
    """
    Add percentiles for deprivation, based on deprivation_score.
    """
    if "deprivation_score" not in imd_lad.columns:
        logger.warning("deprivation_percentiles: missing deprivation_score.")
        imd_lad = imd_lad.copy()
        imd_lad["deprivation_percentile"] = None
        return imd_lad

    df = add_percentiles(
        imd_lad,
        value_col="deprivation_score",
        new_col="deprivation_percentile",
    )
    return df


# ---------------------------------------------------------------------
# 4. Wrapper: full deprivation indicators
# ---------------------------------------------------------------------

def build_deprivation_indicators(imd_lsoa: pd.DataFrame) -> pd.DataFrame:
    """
    High-level wrapper for deprivation indicators.

    Returns LAD-level DataFrame with:
        lad_code
        imd_rank_mean
        imd_decile_mean
        deprivation_score
        deprivation_percentile
    """
    if imd_lsoa is None or imd_lsoa.empty:
        logger.warning("build_deprivation_indicators: empty IMD LSOA input.")
        return pd.DataFrame(
            columns=[
                "lad_code",
                "imd_rank_mean",
                "imd_decile_mean",
                "deprivation_score",
                "deprivation_percentile",
            ]
        )

    lad_agg = imd_aggregate(imd_lsoa)
    lad_score = deprivation_score(lad_agg)
    lad_pct = deprivation_percentiles(lad_score)

    return lad_pct
