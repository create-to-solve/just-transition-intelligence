"""
Ingestion for DfT Road Transport Fuel Consumption (2005–2023).

The DfT workbook contains multiple sheets:
    - Cover / Notes / Metadata sheets
    - One sheet per year: "2005", "2006", ..., "2023"

Each yearly sheet includes LAD-level fuel consumption in ktoe.

We extract the following canonical fields:
    - lad_code
    - lad_name
    - year
    - value  (ktoe)

This loader returns a tidy long-form DataFrame stacked across years.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Updated to v2 dataset name
RAW_PATH = "data/raw/dft_fuel_consumption.xlsx"

# Source column names in yearly sheets
COL_LAD_CODE = "Local Authority Code"
COL_LAD_NAME = "Local Authority [Note 4]"
COL_TOTAL = "Fuel consumption by all vehicles"


def _load_single_year(sheet_name: str, path: str = RAW_PATH) -> pd.DataFrame:
    """
    Load a single year's DfT fuel consumption sheet.

    Parameters
    ----------
    sheet_name : str
        Sheet name (expected to be a year, e.g. '2011').
    path : str
        Path to the Excel workbook.

    Returns
    -------
    pd.DataFrame
        LAD-level records for that year only.
    """
    try:
        year = int(sheet_name)
    except ValueError:
        return pd.DataFrame()  # Ignore non-numeric sheets

    logger.info("Loading DfT sheet for year %s", year)

    # Header for DfT tables typically starts at row index 3
    df = pd.read_excel(path, sheet_name=sheet_name, header=3)

    # Keep rows that have a LAD code
    df = df[df[COL_LAD_CODE].notna()].copy()

    # Keep required columns
    df = df[[COL_LAD_CODE, COL_LAD_NAME, COL_TOTAL]].copy()

    # Standardise column names
    df.columns = ["lad_code", "lad_name", "value_ktoe"]

    # Cleanup whitespace after casting
    df["lad_code"] = df["lad_code"].astype(str).str.strip()
    df["lad_name"] = df["lad_name"].astype(str).str.strip()

    # Rename to canonical name expected downstream
    df.rename(columns={"value_ktoe": "value"}, inplace=True)

    # Add year
    df["year"] = year

    return df


def load_transport() -> pd.DataFrame:
    """
    Load and stack all DfT yearly transport sheets (2005–2023).

    Returns
    -------
    pd.DataFrame
        Long-form LAD–year dataset of fuel consumption in ktoe.
    """
    logger.info("Loading DfT transport dataset from %s", RAW_PATH)

    xl = pd.ExcelFile(RAW_PATH)
    dfs = []

    for sheet in xl.sheet_names:
        try:
            int(sheet)  # only accept numeric sheets
        except ValueError:
            continue

        df_year = _load_single_year(sheet, RAW_PATH)
        if not df_year.empty:
            dfs.append(df_year)

    if not dfs:
        logger.warning("No yearly DfT sheets found.")
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)

    logger.info("DfT transport dataset loaded: %d rows, %d columns", out.shape[0], out.shape[1])

    return out
