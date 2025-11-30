"""
JTIS Pipeline Entry Point
-------------------------

This module defines the high-level execution flow for the
Just Transition Intelligence System (JTIS) pipeline.

It is intentionally simple and linear, reflecting the core
processing stages:

1. Ingestion of raw datasets
2. Harmonisation to LAD–year
3. Indicator construction
4. Composite JTI scoring
5. LAD profile generation

Each step will call functions within the respective modules
(e.g. src/ingestion/, src/harmonisation/, etc.).

No heavy logic should live here.
"""

import logging

# TODO: once implemented, import real functions:
# from src.ingestion.load_emissions import load_emissions
# from src.ingestion.load_population import load_population
# from src.ingestion.load_transport import load_transport
# from src.ingestion.load_imd import load_imd
#
# from src.harmonisation.reshape import harmonise_datasets
# from src.indicators.emissions_indicators import build_emissions_indicators
# from src.indicators.transport_indicators import build_transport_indicators
# from src.indicators.deprivation_indicators import build_deprivation_indicators
# from src.indicators.composite_indicators import combine_indicators
#
# from src.composite.jti_composite import build_jti_composite
# from src.profiles.build_profiles import generate_lad_profiles


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting JTIS pipeline...")

    # ------------------------------------------------------------------
    # 1. INGESTION
    # ------------------------------------------------------------------
    logging.info("Step 1: Ingesting raw datasets...")
    # TODO: Replace placeholder with real function calls
    # emissions = load_emissions()
    # population = load_population()
    # transport = load_transport()
    # imd = load_imd()

    # Placeholder output for Phase 3
    emissions = population = transport = imd = None

    # ------------------------------------------------------------------
    # 2. HARMONISATION
    # ------------------------------------------------------------------
    logging.info("Step 2: Harmonising to LAD–year schema...")
    # TODO: harmonised = harmonise_datasets(emissions, population, transport, imd)
    harmonised = None

    # ------------------------------------------------------------------
    # 3. INDICATORS
    # ------------------------------------------------------------------
    logging.info("Step 3: Building indicators...")
    # TODO: indicators = {
    #     "emissions": build_emissions_indicators(harmonised),
    #     "transport": build_transport_indicators(harmonised),
    #     "deprivation": build_deprivation_indicators(harmonised),
    # }
    indicators = None

    # ------------------------------------------------------------------
    # 4. COMPOSITE JTI SCORE
    # ------------------------------------------------------------------
    logging.info("Step 4: Combining indicators into composite JTI...")
    # TODO: composite_scores = build_jti_composite(indicators)
    composite_scores = None

    # ------------------------------------------------------------------
    # 5. LAD PROFILES
    # ------------------------------------------------------------------
    logging.info("Step 5: Generating LAD profiles...")
    # TODO: profiles = generate_lad_profiles(composite_scores, indicators)
    profiles = None

    logging.info("Pipeline complete.")
    return {
        "harmonised": harmonised,
        "indicators": indicators,
        "composite": composite_scores,
        "profiles": profiles,
    }


if __name__ == "__main__":
    main()
