"""
Composite indicator assembly for JTIS v2.

This module prepares the LAD–year component-level indicators used by
`src/composite/jti_composite.py` for final min–max scaling and JTI score
computation.

It does NOT apply scaling — it only merges:
    - per_capita_emissions
    - transport_mwh_per_capita
    - imd_rank_mean

The input DataFrames are expected to already be harmonised to LAD–year
(except IMD which is per-LSOA and must be aggregated before calling
this function).
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def assemble_composite_inputs(
    ghg_df: pd.DataFrame,
    transport_df: pd.DataFrame,
    deprivation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge emissions, transport and deprivation components into a single
    LAD–year DataFrame.

    Parameters
    ----------
    ghg_df : DataFrame
        Must contain columns:
            - lad_code
            - year
            - per_capita_emissions

    transport_df : DataFrame
        Must contain columns:
            - lad_code
            - year
            - transport_mwh_per_capita

    deprivation_df : DataFrame
        Must contain:
            - lad_code
            - imd_rank_mean
        (Yearless — IMD is a single-year indicator.)

    Returns
    -------
    DataFrame with:
        lad_code, year,
        per_capita_emissions,
        transport_mwh_per_capita,
        imd_rank_mean
    """

    logger.info("Assembling composite inputs...")

    # Base join: emissions ∩ transport on LAD–year
    out = (
        ghg_df[["lad_code", "year", "per_capita_emissions"]]
        .merge(
            transport_df[["lad_code", "year", "transport_mwh_per_capita"]],
            on=["lad_code", "year"],
            how="outer",
        )
    )

    # Deprivation join: single-year, LAD-level
    out = out.merge(
        deprivation_df[["lad_code", "imd_rank_mean"]],
        on="lad_code",
        how="left",
    )

    logger.info("Composite input assembly complete: %d rows, %d columns",
                out.shape[0], out.shape[1])

    return out
