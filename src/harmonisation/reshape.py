"""
Dataset Reshaping and Harmonisation (v2 port)

This file reshapes the three core ingested datasets into LAD–year
long-form tables:

    - DESNZ GHG emissions
    - ONS population (MYEB3)
    - DfT transport fuel consumption

IMD remains at LSOA level here – it is passed through untouched, and
aggregation happens later in the deprivation indicators stage.

Design choice (Option A – preserve v1):
    - No single “master” LAD–year table is built here.
    - Instead, each dataset is reshaped and validated separately.
    - Merges happen later in indicator and composite stages.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger
from src.harmonisation.geography import normalise_lad_code
from src.utils.units import ktoe_to_mwh
from src.harmonisation.validation import (
    validate_schema,
    validate_geography,
    validate_years,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _detect_year_column(df: pd.DataFrame, dataset_name: str) -> Optional[str]:
    """
    Try to detect a column representing 'year'.
    """
    # Literal 'year'
    for col in df.columns:
        if col.lower() == "year":
            return col

    # Single numeric-year column (e.g., 2023)
    year_like = []
    for col in df.columns:
        try:
            v = int(str(col))
            if 1900 <= v <= 2100:
                year_like.append(col)
        except ValueError:
            continue

    if len(year_like) == 1:
        return year_like[0]

    if not year_like:
        logger.warning("%s reshape: no explicit year column detected.", dataset_name)
    else:
        logger.info(
            "%s reshape: multiple year-like columns found %s – caller must decide.",
            dataset_name,
            year_like,
        )
    return None


def _safe_numeric_columns(df: pd.DataFrame, exclude: Tuple[str, ...]) -> List[str]:
    """
    Return numeric column names excluding specified ones.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


# ---------------------------------------------------------------------
# 1. DESNZ GHG emissions
# ---------------------------------------------------------------------

def reshape_emissions(df: pd.DataFrame, dataset_name: str = "DESNZ GHG") -> pd.DataFrame:
    """
    Reshape DESNZ GHG emissions to LAD–year totals.
    """
    if df is None or df.empty:
        logger.warning("Emissions reshape: empty input frame.")
        return pd.DataFrame(columns=["lad_code", "lad_name", "year", "emissions_tonnes"])

    df = normalise_lad_code(df, dataset_name=dataset_name)
    year_col = _detect_year_column(df, dataset_name)

    if year_col is None or "lad_code" not in df.columns:
        logger.warning("Emissions reshape: cannot proceed without lad_code/year.")
        return pd.DataFrame(columns=["lad_code", "lad_name", "year", "emissions_tonnes"])

    numeric_cols = _safe_numeric_columns(df, exclude=("year", year_col, "lad_code"))
    if not numeric_cols:
        logger.warning("Emissions reshape: no numeric columns to aggregate.")
        return pd.DataFrame(columns=["lad_code", "lad_name", "year", "emissions_tonnes"])

    group_cols = ["lad_code", year_col] + (["lad_name"] if "lad_name" in df.columns else [])
    agg = (
        df[group_cols + numeric_cols]
        .groupby(group_cols, dropna=False)[numeric_cols]
        .sum()
        .reset_index()
    )

    # Collapse all numeric columns into one emissions_tonnes column
    agg["emissions_tonnes"] = agg[numeric_cols].sum(axis=1)

    cols = ["lad_code"]
    if "lad_name" in agg.columns:
        cols.append("lad_name")
    cols.extend(["year", "emissions_tonnes"])

    result = agg.rename(columns={year_col: "year"})[cols]

    logger.info("Emissions reshape: %d → %d LAD–year rows.", len(df), len(result))
    return result


# Alias for compatibility
reshape_desnz_ghg = reshape_emissions


# ---------------------------------------------------------------------
# 2. ONS population
# ---------------------------------------------------------------------

def reshape_population(df: pd.DataFrame, dataset_name: str = "ons_population") -> pd.DataFrame:
    """
    Reshape ONS MYEB3 wide population dataset to long LAD–year format.
    """
    if df is None or df.empty:
        logger.warning("Population reshape: empty input frame.")
        return pd.DataFrame(columns=["lad_code", "lad_name", "year", "population"])

    df = normalise_lad_code(df, dataset_name=dataset_name)

    pop_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("population_")]
    if not pop_cols:
        logger.warning("Population reshape: no population_* columns found.")
        return pd.DataFrame(columns=["lad_code", "lad_name", "year", "population"])

    id_vars = ["lad_code"] + (["lad_name"] if "lad_name" in df.columns else [])

    long_df = df.melt(
        id_vars=id_vars,
        value_vars=pop_cols,
        var_name="year",
        value_name="population",
    )

    # Extract numeric year from 'population_2011'
    long_df["year"] = long_df["year"].str.replace("population_", "", regex=False)
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["population"] = pd.to_numeric(long_df["population"], errors="coerce")

    before = len(long_df)
    long_df = long_df.dropna(subset=["year"])
    after = len(long_df)

    logger.info("Population reshape: %d → %d LAD–year rows.", before, after)

    cols = ["lad_code"]
    if "lad_name" in long_df.columns:
        cols.append("lad_name")
    cols.extend(["year", "population"])

    return long_df[cols]


# ---------------------------------------------------------------------
# 3. DfT transport fuel consumption
# ---------------------------------------------------------------------

def reshape_transport(
    df: pd.DataFrame,
    dataset_name: str = "DfT fuel consumption",
    default_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reshape DfT fuel consumption to LAD–year format (MWh).
    """
    if df is None or df.empty:
        logger.warning("Transport reshape: empty input frame.")
        return pd.DataFrame(columns=["lad_code", "lad_name", "year", "transport_mwh"])

    df = normalise_lad_code(df, dataset_name=dataset_name)
    year_col = _detect_year_column(df, dataset_name)

    if year_col is None:
        if default_year is None:
            logger.warning("Transport reshape: no year and no default_year.")
            return pd.DataFrame(columns=["lad_code", "lad_name", "year", "transport_mwh"])
        df["year"] = default_year
        year_col = "year"

    numeric_cols = _safe_numeric_columns(df, exclude=("year", year_col, "lad_code"))
    if not numeric_cols:
        logger.warning("Transport reshape: no numeric columns to aggregate.")
        return pd.DataFrame(columns=["lad_code", "lad_name", "year", "transport_mwh"])

    group_cols = ["lad_code", year_col] + (["lad_name"] if "lad_name" in df.columns else [])
    agg = (
        df[group_cols + numeric_cols]
        .groupby(group_cols, dropna=False)[numeric_cols]
        .sum()
        .reset_index()
    )

    agg["transport_ktoe"] = agg[numeric_cols].sum(axis=1)
    agg["transport_mwh"] = ktoe_to_mwh(agg["transport_ktoe"])

    cols = ["lad_code"]
    if "lad_name" in agg.columns:
        cols.append("lad_name")
    cols.extend(["year", "transport_mwh"])

    result = agg.rename(columns={year_col: "year"})[cols]

    logger.info("Transport reshape: %d → %d LAD–year rows.", len(df), len(result))
    return result


# ---------------------------------------------------------------------
# 4. Orchestrator: harmonise_datasets (Option A style)
# ---------------------------------------------------------------------

def harmonise_datasets(
    ghg_raw: pd.DataFrame,
    transport_raw: pd.DataFrame,
    population_raw: pd.DataFrame,
    imd_raw: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Harmonise shapes (wide → long where needed) and run basic checks.

    Mirrors the behaviour of v1 run_harmonisation, but as a pure
    function that returns a dict of harmonised tables.

    Returns
    -------
    dict with keys:
        - "ghg_long"
        - "transport_long"
        - "population_long"
        - "imd_lsoa"
    """
    logger.info("=== Stage 2: Harmonisation (reshape + basic checks) ===")

    # 4.1 Reshape
    ghg_long = reshape_emissions(ghg_raw, dataset_name="desnz_ghg")
    transport_long = reshape_transport(transport_raw, dataset_name="dft_transport")
    population_long = reshape_population(population_raw, dataset_name="ons_population")
    imd_lsoa = imd_raw.copy()

    # 4.2 Basic validations (non-fatal)

    # DESNZ
    _ = validate_schema(ghg_long, ["lad_code", "year"], "desnz_ghg")
    _ = validate_geography(ghg_long, "desnz_ghg")
    _ = validate_years(ghg_long, "desnz_ghg")

    # DfT transport
    _ = validate_schema(transport_long, ["lad_code", "year"], "dft_transport")
    _ = validate_geography(transport_long, "dft_transport")
    _ = validate_years(transport_long, "dft_transport")

    # ONS population
    _ = validate_schema(
        population_long,
        ["lad_code", "year", "population"],
        "ons_population",
    )
    _ = validate_geography(population_long, "ons_population")
    _ = validate_years(population_long, "ons_population")

    # IMD – check geography if lad_code present
    if "lad_code" in imd_lsoa.columns:
        _ = validate_geography(imd_lsoa, "imd_2019")
    else:
        logger.warning(
            "IMD LSOA dataset missing 'lad_code'; LAD linkage will fail later."
        )

    logger.info("Harmonisation complete.")
    return {
        "ghg_long": ghg_long,
        "transport_long": transport_long,
        "population_long": population_long,
        "imd_lsoa": imd_lsoa,
    }
