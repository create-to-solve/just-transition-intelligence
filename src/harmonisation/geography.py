"""
Geographic Harmonisation Utilities

This module provides LAD code/name normalisation and optional lookup
enrichment. It is a direct, clean port of the v1 logic, with only
minor structural edits for the v2 repository layout.

Key features:
    - Automatically detects LAD code and name columns from a list of
      known variants (e.g. LAD22CD, ladcode23, Local Authority Code).
    - Normalises LAD codes to uppercase strings with no whitespace.
    - Optionally attaches canonical LAD metadata by joining a lookup
      table (src.utils.lad_lookup).

No boundary-change transformations are introduced here unless present
in the original logic.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Candidate column names for LAD code/name fields
# ---------------------------------------------------------------------

__LAD_CODE_CANDIDATES: Iterable[str] = (
    "lad_code",
    "ladcode23",
    "LADCD",
    "LAD21CD",
    "LAD22CD",
    "LA Code",
    "Local Authority Code",
    "Local_Authority_Code",
    "Area Codes",
    "AREACD",
    "Geography code",
    "GEOGRAPHY_CODE",
    "Code",
)

__LAD_NAME_CANDIDATES: Iterable[str] = (
    "lad_name",
    "laname23",
    "LADNM",
    "LAD21NM",
    "LAD22NM",
    "LA Name",
    "Local Authority Name",
    "Local_Authority_Name",
    "Area Names",
    "AREANM",
    "Name",
)

# ---------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------

def _find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """
    Return the first candidate column present in df, or None if none exist.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def normalise_lad_code(
    df: pd.DataFrame,
    code_col: Optional[str] = None,
    name_col: Optional[str] = None,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """
    Ensure df has standard 'lad_code' (and optionally 'lad_name') columns.

    Logic (identical to v1):
        - If code_col is provided, use that.
        - Otherwise search a list of known LAD code column variants.
        - Clean codes: cast to string, strip whitespace, uppercase.
        - Identify LAD name column using similar candidate list.
        - Return a *copy* to avoid mutating upstream inputs.

    If no LAD code column is found:
        - Logs a warning.
        - Creates 'lad_code' filled with NA so that validation in the
          next layer will catch the issue cleanly.
    """
    if df is None or df.empty:
        logger.warning("normalise_lad_code: empty frame for %s", dataset_name)
        return df

    df = df.copy()

    # Resolve LAD code column
    if code_col and code_col in df.columns:
        src_code_col = code_col
    else:
        src_code_col = _find_first_column(df, __LAD_CODE_CANDIDATES)

    if src_code_col is None:
        logger.warning(
            "Geography: no LAD code column found for %s. "
            "Creating empty 'lad_code'.",
            dataset_name,
        )
        df["lad_code"] = pd.NA
    else:
        if src_code_col != "lad_code":
            df.rename(columns={src_code_col: "lad_code"}, inplace=True)

        df["lad_code"] = (
            df["lad_code"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # Resolve LAD name column (optional)
    if name_col and name_col in df.columns:
        src_name_col = name_col
    else:
        src_name_col = _find_first_column(df, __LAD_NAME_CANDIDATES)

    if src_name_col is None:
        logger.info(
            "Geography: no LAD name column found for %s. Proceeding without 'lad_name'.",
            dataset_name,
        )
    else:
        if src_name_col != "lad_name":
            df.rename(columns={src_name_col: "lad_name"}, inplace=True)

    return df


def apply_lad_lookup(
    df: pd.DataFrame,
    lookup_df: Optional[pd.DataFrame] = None,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """
    Attach canonical LAD metadata using a lookup table.

    Behaviour (preserved from v1):
        - If lookup_df is provided, use it directly.
        - If not, attempt to load it from src.utils.lad_lookup.get_lad_lookup().
        - Non-fatal failures: logs warning and returns df unchanged.
        - Resolves overlapping columns safely.
    """
    if df is None or df.empty:
        logger.warning("apply_lad_lookup: empty frame for %s", dataset_name)
        return df

    if "lad_code" not in df.columns:
        logger.warning(
            "apply_lad_lookup: 'lad_code' missing for %s – cannot join lookup.",
            dataset_name,
        )
        return df

    # Try to load lookup if not provided
    if lookup_df is None:
        try:
            from src.utils.lad_lookup import get_lad_lookup  # type: ignore
            lookup_df = get_lad_lookup()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "apply_lad_lookup: could not load LAD lookup for %s (%s). "
                "Proceeding without enrichment.",
                dataset_name,
                exc,
            )
            return df

    if lookup_df is None or lookup_df.empty:
        logger.warning(
            "apply_lad_lookup: empty lookup for %s – no enrichment applied.",
            dataset_name,
        )
        return df

    if "lad_code" not in lookup_df.columns:
        logger.warning(
            "apply_lad_lookup: lookup missing 'lad_code' for %s – no join.",
            dataset_name,
        )
        return df

    # Avoid collisions except lad_code
    lookup_cols = [c for c in lookup_df.columns if c != "lad_code"]
    overlap = set(df.columns).intersection(lookup_cols)
    if overlap:
        logger.info(
            "apply_lad_lookup: dropping overlapping columns from lookup for %s: %s",
            dataset_name,
            sorted(overlap),
        )
        lookup_df = lookup_df.drop(columns=list(overlap))

    merged = df.merge(lookup_df, on="lad_code", how="left")
    logger.info(
        "apply_lad_lookup: merged LAD lookup for %s (%d → %d rows).",
        dataset_name,
        len(df),
        len(merged),
    )
    return merged
