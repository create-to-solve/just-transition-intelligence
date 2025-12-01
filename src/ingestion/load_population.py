"""
Ingestion for ONS Mid-Year Population Estimates (2011–2024).

This loader reads the MYEB3 sheet from the ONS population workbook.

Expected fields include:
    - ladcode23
    - laname23
    - country
    - population_2011 ... population_2024
    - additional births/deaths columns (ignored)

We only ingest the raw workbook here. Harmonisation into LAD–year
format will happen in the harmonisation layer.
"""

from __future__ import annotations

import os
import pandas as pd

from src.utils.logger import get_logger
# Note: read_table not used here because this is a structured Excel workbook.
# Using pandas directly is appropriate.
logger = get_logger(__name__)


def load_population(
    path: str = "data/raw/ons_population.xlsx",
    sheet_name: str = "MYEB3",
) -> pd.DataFrame:
    """
    Load ONS mid-year population estimates (LAD totals) from MYEB3.

    Parameters
    ----------
    path : str
        Path to the Excel workbook (defaults to v2 raw data location).
    sheet_name : str
        Sheet containing LAD totals (default "MYEB3").

    Returns
    -------
    pd.DataFrame
        Raw ONS population dataset as loaded from Excel. No column
        renaming or LAD–year reshaping is done here.
    """
    full_path = os.path.abspath(path)
    logger.info("Loading ONS population dataset from %s (sheet=%s)", full_path, sheet_name)

    df = pd.read_excel(full_path, sheet_name=sheet_name, header=1)

    logger.info(
        "ONS population dataset loaded: %d rows, %d columns",
        df.shape[0], df.shape[1]
    )

    return df
