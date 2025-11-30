"""
Energy and emissions unit conversions.

Direct port of v1 logic with updated logger import.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

NumberLike = Union[int, float, np.number, pd.Series, pd.Index]

# Conversion factors
KTOE_TO_MWH_FACTOR = 11.63 * 1000.0     # 1 ktoe = 11.63 MWh × 1000
KTCO2E_TO_TONNES_FACTOR = 1000.0        # 1 ktCO2e = 1000 tonnes


def ktoe_to_mwh(x: NumberLike) -> NumberLike:
    """
    Convert thousand tonnes of oil equivalent (ktoe) to MWh.
    """
    try:
        arr = pd.to_numeric(x, errors="coerce")
        if isinstance(x, (pd.Series, pd.Index)) and arr.isna().any():
            logger.warning("ktoe_to_mwh: some values could not be converted to numeric.")
        return arr * KTOE_TO_MWH_FACTOR
    except Exception as exc:
        logger.warning("ktoe_to_mwh: conversion failed (%s). Returning input.", exc)
        return x


def ktco2e_to_tonnes(x: NumberLike) -> NumberLike:
    """
    Convert kilotonnes of CO2e (ktCO2e) to tonnes CO2e.
    """
    try:
        arr = pd.to_numeric(x, errors="coerce")
        if isinstance(x, (pd.Series, pd.Index)) and arr.isna().any():
            logger.warning("ktco2e_to_tonnes: some values could not be converted to numeric.")
        return arr * KTCO2E_TO_TONNES_FACTOR
    except Exception as exc:
        logger.warning("ktco2e_to_tonnes: conversion failed (%s). Returning input.", exc)
        return x
