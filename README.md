# Local-authority evidence system for place-based transition analysis

An offline, reproducible framework for analysing UK local-authority transition readiness using four core datasets:

- DESNZ greenhouse gas emissions
- DfT transport energy/fuel use
- ONS population
- MHCLG IMD 2019**

Outputs harmonised indicators, composite scores, clusters, and a dashboard designed for councils and researchers.

Per-capita emissions, transport energy use, IMD mean rank, LAD × Year harmonised tables.

## Composite Just Transition Index (JTI):
Three components (emissions, transport, deprivation) with national percentiles and ranks.

## Place-based clusters:
Each LAD is assigned to a cluster describing typical transition contexts.

## Interactive dashboard:
Executive Summary, Components, Trends, Comparisons, Clusters.

## System Architecture

Raw data → Ingestion → Harmonisation → Indicators → Composite (JTI) → Profiles + Clusters → Dashboard

## Repository Structure

1. src/ingestion – raw dataset loaders
2. src/harmonisation – LAD/year alignment, validation
3. src/indicators – emissions, transport, deprivation indicators
4. src/composite – JTI and weighting
5. src/profiles – LAD profiles, cluster assignment
6. src/dashboard – Streamlit app
7. config/ – dataset and settings YAML
8. outputs/ – pipeline results (jti_scores.csv, lad_profile_summary.csv)
9. notebooks/ – exploration and validation

## Usage

- Create your environment and install dependencies
- Place offline datasets as configured in config/datasets.yaml
- Run the pipeline using run_pipeline.sh
- Launch the dashboard using streamlit run src/dashboard/app.py

Outputs appear in the outputs/ folder.
