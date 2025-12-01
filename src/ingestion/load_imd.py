"""
Ingestion for IMD 2019 using the known workbook structure.

Expected source columns include:
    - LSOA code (2011)
    - LSOA name (2011)
    - Local Authority District code (2019)
    - Local Authority District name (2019)
    - Index of Multiple Deprivation (IMD) Rank
    - Index of Multiple Deprivation (IMD) Decile

This ingestion step only cleans column names and loads the IMD2019 sheet.
Aggregation to LAD–year occurs in the harmonisation layer.
"""

import os
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_PATH = "data/raw/imd_2019.xlsx"


def load_imd() -> pd.DataFrame:
    """
    Load IMD 2019 (LSOA-level) with clean, standardised column names.

    Returns
    -------
    pd.DataFrame
        Columns:
            - lsoa_code
            - lsoa_name
            - lad_code
            - lad_name
            - imd_rank
            - imd_decile
    """
    full_path = os.path.abspath(RAW_PATH)
    logger.info("Loading IMD dataset from %s (sheet=IMD2019)", full_path)

    # Read explicitly from the "IMD2019" sheet
    df = pd.read_excel(full_path, sheet_name="IMD2019")

    # Standardise column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Map raw column names → canonical internal schema
    rename_map = {
        "lsoa_code_(2011)": "lsoa_code",
        "lsoa_name_(2011)": "lsoa_name",
        "local_authority_district_code_(2019)": "lad_code",
        "local_authority_district_name_(2019)": "lad_name",
        "index_of_multiple_deprivation_(imd)_rank": "imd_rank",
        "index_of_multiple_deprivation_(imd)_decile": "imd_decile",
    }
    df = df.rename(columns=rename_map)

    logger.info("IMD dataset loaded: %d rows, %d columns", df.shape[0], df.shape[1])
    return df
