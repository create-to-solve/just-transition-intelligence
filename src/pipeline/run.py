"""
JTIS v2 Pipeline Entrypoint

Runs the full offline processing workflow:

    1. Ingestion (4 datasets)
    2. Harmonisation (LAD–year long tables)
    3. Indicators (emissions, transport, deprivation)
    4. Composite inputs and JTI score (v1 logic)
    5. LAD profile summary (v1 logic)
    6. Write minimal outputs (Option A)

Run with:
    python -m src.pipeline.run
"""

from __future__ import annotations

import os
import pandas as pd

from src.utils.logger import get_logger

# Ingestion
from src.ingestion.load_emissions import load_emissions
from src.ingestion.load_transport import load_transport
from src.ingestion.load_population import load_population
from src.ingestion.load_imd import load_imd

# Harmonisation
from src.harmonisation.reshape import (
    reshape_emissions,
    reshape_transport,
    reshape_population,
    harmonise_datasets,
)

# Indicators
from src.indicators.emissions_indicators import (
    per_capita_emissions,
    yoy_change as ghg_yoy_change,
)
from src.indicators.transport_indicators import (
    transport_mwh_per_capita,
    transport_yoy_pct,
)
from src.indicators.deprivation_indicators import build_deprivation_indicators

# Composite + JTI
from src.composite.jti_composite import build_jti_scores

# Profiles
from src.profiles.build_profiles import build_lad_profiles


logger = get_logger(__name__)

DATA_PROCESSED = "data/processed"
OUTPUTS = "outputs"


def ensure_dirs():
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    os.makedirs(OUTPUTS, exist_ok=True)


def main():

    logger.info("=== JTIS v2 Pipeline Starting ===")

    ensure_dirs()

    # --------------------------------------------------
    # 1. INGESTION
    # --------------------------------------------------
    ghg_raw = load_emissions()
    transport_raw = load_transport()
    population_raw = load_population()
    imd_raw = load_imd()

    # --------------------------------------------------
    # 2. HARMONISATION
    # --------------------------------------------------
    harm = harmonise_datasets(
        ghg_raw,
        transport_raw,
        population_raw,
        imd_raw,
    )

    ghg_long = harm["ghg_long"]
    transport_long = harm["transport_long"]
    population_long = harm["population_long"]
    imd_lsoa = harm["imd_lsoa"]

    # Save harmonised LAD–year tables
    ghg_long.to_csv(f"{DATA_PROCESSED}/ghg_long.csv", index=False)
    transport_long.to_csv(f"{DATA_PROCESSED}/transport_long.csv", index=False)
    population_long.to_csv(f"{DATA_PROCESSED}/population_long.csv", index=False)

    # Deprivation: aggregate LSOA → LAD
    imd_lad = build_deprivation_indicators(imd_lsoa)
    imd_lad.to_csv(f"{DATA_PROCESSED}/imd_lad.csv", index=False)

    # --------------------------------------------------
    # 3. INDICATORS
    # --------------------------------------------------

    # Emissions indicators
    ghg_ind = ghg_long.merge(
        population_long, on=["lad_code", "year"], how="left"
    )
    ghg_ind = per_capita_emissions(ghg_ind)
    ghg_ind = ghg_yoy_change(ghg_ind)

    # Transport indicators
    transport_ind = transport_long.merge(
        population_long, on=["lad_code", "year"], how="left"
    )
    transport_ind = transport_mwh_per_capita(transport_ind)
    transport_ind = transport_yoy_pct(transport_ind)

    # --------------------------------------------------
    # 4. COMPOSITE + JTI
    # --------------------------------------------------
    jti_df = build_jti_scores(
        ghg_df=ghg_ind,
        transport_df=transport_ind,
        deprivation_df=imd_lad,
    )

    # --------------------------------------------------
    # Add LAD name (needed for dashboard)
    # --------------------------------------------------
    lad_name_map = population_long[["lad_code", "lad_name"]].drop_duplicates()
    jti_df = jti_df.merge(lad_name_map, on="lad_code", how="left")

    # --------------------------------------------------
    # Add last-year component fields (v1 behaviour)
    # --------------------------------------------------
    def extract_last_components(df):
        out = []
        for lad_code, grp in df.groupby("lad_code"):
            grp = grp.sort_values("year")
            last = grp.iloc[-1]
            out.append({
                "lad_code": lad_code,
                "component_emissions_last": last.get("per_capita_emissions_scaled", None),
                "component_transport_last": last.get("transport_scaled", None),
                "component_deprivation_last": last.get("deprivation_scaled", None),
            })
        return pd.DataFrame(out)

    last_components = extract_last_components(jti_df)
    jti_df = jti_df.merge(last_components, on="lad_code", how="left")

    jti_df.to_csv(f"{OUTPUTS}/jti_scores.csv", index=False)

    # --------------------------------------------------
    # 5. LAD PROFILES
    # --------------------------------------------------
    lad_profiles = build_lad_profiles(jti_df)
    lad_profiles.to_csv(f"{OUTPUTS}/lad_profile_summary.csv", index=False)

    logger.info("=== JTIS v2 Pipeline Complete ===")


if __name__ == "__main__":
    main()
