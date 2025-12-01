"""
Emissions indicators (v2 port, behaviour identical to v1).

Indicators derived from harmonised DESNZ GHG data.

Expected input columns:
    lad_code, year, emissions_tonnes, population

Provides:
    - per_capita_emissions
    - yoy_change (emissions_yoy_pct)
    - emissions_index (base = min year per LAD unless specified)
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# 1. Per-capita emissions
# ---------------------------------------------------------------------

def per_capita_emissions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CO2e per-capita emissions.

    Requires:
        emissions_tonnes
        population
    """
    required = {"emissions_tonnes", "population"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("per_capita_emissions: missing columns: %s", missing)
        return df

    df = df.copy()
    df["per_capita_emissions"] = df["emissions_tonnes"] / df["population"]
    return df


# ---------------------------------------------------------------------
# 2. Year-on-year percentage change
# ---------------------------------------------------------------------

def yoy_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute year-on-year % change in emissions for each LAD.

    Produces:
        emissions_yoy_pct
    """
    if not {"lad_code", "year", "emissions_tonnes"}.issubset(df.columns):
        logger.warning("yoy_change: missing required columns.")
        return df

    df = df.copy().sort_values(["lad_code", "year"])
    df["emissions_yoy_pct"] = df.groupby("lad_code")["emissions_tonnes"].pct_change() * 100
    return df


# ---------------------------------------------------------------------
# 3. Index relative to base year
# ---------------------------------------------------------------------

def emissions_index(df: pd.DataFrame, base_year: int | None = None) -> pd.DataFrame:
    """
    Compute an index relative to a base year (default = earliest available per LAD).

    Produces:
        emissions_index (base = 100)
    """
    if not {"lad_code", "year", "emissions_tonnes"}.issubset(df.columns):
        logger.warning("emissions_index: missing required columns.")
        return df

    df = df.copy()

    if base_year is None:
        # LAD-specific base years
        base = (
            df.sort_values(["lad_code", "year"])
              .groupby("lad_code")
              .first()["emissions_tonnes"]
        )
        df = df.join(base.rename("base_emissions"), on="lad_code")
    else:
        # Global base year
        base_df = df[df["year"] == base_year].set_index("lad_code")["emissions_tonnes"]
        df = df.join(base_df.rename("base_emissions"), on="lad_code")

    df["emissions_index"] = (df["emissions_tonnes"] / df["base_emissions"]) * 100
    return df
