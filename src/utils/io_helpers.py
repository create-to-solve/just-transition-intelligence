"""
I/O helper functions for reading CSV and Excel files,
plus a unified read_table() helper for ingestion modules.
"""

from __future__ import annotations

import os
import pandas as pd


def read_csv(path: str, **kwargs):
    """Read a CSV file with UTF-8 and safe defaults."""
    return pd.read_csv(path, **kwargs)


def read_excel(path: str, **kwargs):
    """Read an Excel file."""
    return pd.read_excel(path, **kwargs)


def read_table(path: str, **kwargs):
    """
    Unified read helper used in ingestion modules.

    Behaviour matches v1:
        - .csv   → read_csv
        - .xls/.xlsx → read_excel
        - anything else → error
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path, **kwargs)

    if ext in [".xls", ".xlsx"]:
        return pd.read_excel(path, **kwargs)

    raise ValueError(f"Unsupported file extension for read_table: {ext}")
