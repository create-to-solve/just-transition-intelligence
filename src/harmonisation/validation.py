"""
Validation and QA for harmonised data.

These helpers perform basic consistency checks:
    - schema: required columns present
    - geography: lad_code existence and completeness
    - years: year column present, numeric, and (optionally) unique per LAD
"""

from __future__ import annotations

from typing import List, Sequence

import pandas as pd

from src.utils.logger import get_logger
from src.harmonisation.geography import normalise_lad_code

logger = get_logger(__name__)


def validate_schema(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    dataset_name: str,
) -> bool:
    """
    Check that required_columns are present in df.

    Logs a warning listing missing columns if any are absent.
    Returns True if schema is valid, False otherwise.
    """
    if df is None or df.empty:
        logger.warning("Schema validation: %s is empty.", dataset_name)
        return False

    missing: List[str] = [c for c in required_columns if c not in df.columns]
    if missing:
        logger.warning(
            "Schema validation (%s): missing columns: %s",
            dataset_name,
            missing,
        )
        return False

    logger.info("Schema validation (%s): OK.", dataset_name)
    return True


def validate_geography(df: pd.DataFrame, dataset_name: str) -> bool:
    """
    Validate LAD geography for a dataset.

    - Ensures lad_code is present (normalising if needed).
    - Logs count of rows with missing lad_code.
    """
    if df is None or df.empty:
        logger.warning("Geography validation: %s is empty.", dataset_name)
        return False

    if "lad_code" not in df.columns:
        df = normalise_lad_code(df, dataset_name=dataset_name)

    if "lad_code" not in df.columns:
        logger.warning(
            "Geography validation: %s still missing 'lad_code' after normalisation.",
            dataset_name,
        )
        return False

    missing_mask = df["lad_code"].isna() | (df["lad_code"].astype(str).str.strip() == "")
    missing_count = missing_mask.sum()

    if missing_count > 0:
        logger.warning(
            "Geography validation: %s has %d rows with missing lad_code.",
            dataset_name,
            missing_count,
        )
        return False

    logger.info("Geography validation: %s OK.", dataset_name)
    return True


def validate_years(
    df: pd.DataFrame,
    dataset_name: str,
    allow_duplicates: bool = False,
) -> bool:
    """
    Validate presence and sanity of 'year' column.

    - Checks that 'year' exists.
    - Coerces to integer where possible.
    - If allow_duplicates is False, logs a warning if duplicate
      (lad_code, year) pairs exist.

    Returns True if the year dimension passes checks, False otherwise.
    """
    if df is None or df.empty:
        logger.warning("Year validation: %s is empty.", dataset_name)
        return False

    if "year" not in df.columns:
        logger.warning(
            "Year validation: missing 'year' column in %s.", dataset_name
        )
        return False

    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    if df["year"].isna().any():
        logger.warning(
            "Year validation: some non-numeric or missing years in %s.", dataset_name
        )

    if not allow_duplicates and "lad_code" in df.columns:
        dup = df.duplicated(subset=["lad_code", "year"], keep=False)
        if dup.any():
            dup_counts = (
                df.loc[dup, ["lad_code", "year"]]
                .value_counts()
                .to_dict()
            )
            logger.warning(
                "Year validation: duplicate (lad_code, year) combinations in %s: %s",
                dataset_name,
                dup_counts,
            )
            # Still return True – this is a warning, not fatal.
            return True

    logger.info("Year validation: %s OK.", dataset_name)
    return True
