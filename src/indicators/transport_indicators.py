"""
Transport indicators (v2 port, behaviour identical to v1).

Indicators derived from reshaped DfT transport dataset.

Expected columns:
    lad_code, lad_name, year, transport_mwh, population

Provides:
    - transport_mwh_per_capita
    - transport_yoy_pct
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# 1. Per-capita transport energy
# ---------------------------------------------------------------------

def transport_mwh_per_capita(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transport MWh per capita.

    Requires:
        transport_mwh
        population
    """
    df = df.copy()

    if "transport_mwh" not in df.columns or "population" not in df.columns:
        logger.warning("transport_mwh_per_capita: missing required columns.")
        df["transport_mwh_per_capita"] = None
        return df

    df["transport_mwh_per_capita"] = df["transport_mwh"] / df["population"]
    return df


# ---------------------------------------------------------------------
# 2. Year-on-year % change in transport energy
# ---------------------------------------------------------------------

def transport_yoy_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute YoY percentage change in transport MWh.

    Produces:
        transport_yoy_pct
    """
    required = {"lad_code", "year", "transport_mwh"}
    if not required.issubset(df.columns):
        logger.warning("transport_yoy_pct: missing required columns.")
        df = df.copy()
        df["transport_yoy_pct"] = None
        return df

    df = df.copy().sort_values(["lad_code", "year"])
    df["transport_yoy_pct"] = df.groupby("lad_code")["transport_mwh"].pct_change() * 100
    return df
