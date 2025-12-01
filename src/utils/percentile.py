from __future__ import annotations

"""
Percentile utilities.

Provides a single helper:
    add_percentiles(df, value_col, new_col="percentile")

Adds a 0–100 percentile rank for each value in the specified column.
"""

import pandas as pd
import numpy as np


def add_percentiles(
    df: pd.DataFrame,
    value_col: str,
    new_col: str = "percentile",
) -> pd.DataFrame:
    """
    Add percentile ranks (0–100) for the given value column.

    Behaviour identical to v1:
        - Higher value → higher percentile
        - Missing/invalid values handled gracefully
    """
    out = df.copy()

    if value_col not in out.columns:
        out[new_col] = np.nan
        return out

    series = pd.to_numeric(out[value_col], errors="coerce")

    # Percentiles: rank normalized * 100
    out[new_col] = series.rank(method="min", pct=True) * 100

    return out
