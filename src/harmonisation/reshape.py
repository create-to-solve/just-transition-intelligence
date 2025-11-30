"""
Dataset Reshaping and Harmonisation

This module turns four ingested datasets into a unified LAD–year shape:
    - DESNZ emissions
    - DfT transport
    - ONS population
    - IMD (requires LSOA→LAD aggregation)

At this phase, we only define the function signatures.
Actual logic will be ported from the old repo incrementally.
"""

def harmonise_datasets(emissions, transport, population, imd):
    """
    Harmonise all ingested datasets into a consistent LAD–year schema.

    Parameters
    ----------
    emissions : pd.DataFrame
    transport : pd.DataFrame
    population : pd.DataFrame
    imd : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Harmonised LAD–year dataset.

    TODO:
        - Bring in old harmonisation logic.
        - Apply LAD normalisation.
        - Aggregate IMD from LSOA to LAD.
        - Construct merged LAD–year index.
    """
    raise NotImplementedError("Dataset harmonisation to be implemented.")
