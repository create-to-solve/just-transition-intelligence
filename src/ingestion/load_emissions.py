"""
Ingestion for DESNZ Local Authority GHG emissions, 2005–2023.

The raw file is already long-form and has the following key columns:

    Local Authority Code              → lad_code
    Local Authority                   → lad_name
    Calendar Year                     → year
    Territorial emissions (kt CO2e)   → emissions_kt_co2e
    Mid-year Population (thousands)   → population_thousands

We standardise and return a tidy long-form table with canonical names.
"""

import pandas as pd
from src.utils.io_helpers import read_table
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_PATH = "data/raw/desnz_ghg_emissions.csv"  # matches v2 dataset naming


def load_emissions() -> pd.DataFrame:
    """
    Load and standardise the DESNZ GHG emissions dataset.

    Returns
    -------
    pd.DataFrame
        Tidy LAD–year dataset with:
        - lad_code
        - lad_name
        - year
        - emissions_tonnes
        - population
    """
    logger.info("Loading DESNZ GHG dataset from %s", RAW_PATH)

    df = read_table(RAW_PATH)

    # Standardise column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Rename key fields from source conventions → canonical names
    rename_map = {
        "local_authority_code": "lad_code",
        "local_authority": "lad_name",
        "calendar_year": "year",
        "territorial_emissions_(kt_co2e)": "emissions_kt_co2e",
        "mid-year_population_(thousands)": "population_thousands",
    }
    df = df.rename(columns=rename_map)

    # Convert emissions to absolute tonnes
    if "emissions_kt_co2e" in df.columns:
        df["emissions_tonnes"] = df["emissions_kt_co2e"] * 1000
    else:
        logger.warning("DESNZ: emissions_kt_co2e column not found.")

    # Convert population to absolute people count
    if "population_thousands" in df.columns:
        df["population"] = df["population_thousands"] * 1000
    else:
        logger.warning("DESNZ: population_thousands column not found.")

    # Keep canonical columns only
    keep_cols = [
        "lad_code",
        "lad_name",
        "year",
        "emissions_tonnes",
        "population",
    ]
    df = df[keep_cols].dropna(subset=["lad_code", "year"])

    logger.info(
        "DESNZ GHG dataset loaded: %d rows, %d columns",
        df.shape[0],
        df.shape[1],
    )
    return df
