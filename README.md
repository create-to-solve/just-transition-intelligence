# **Just Transition Intelligence System (JTIS)
## Local-authority evidence system for place-based transition analysis
Version 2 (2025)

The Just Transition Intelligence System (JTIS) is an offline, reproducible framework for analysing UK local-authority transition readiness using four core datasets:

**DESNZ greenhouse gas emissions

DfT transport energy/fuel use

ONS population

MHCLG IMD 2019**

JTIS outputs harmonised indicators, composite scores, clusters, and a dashboard designed for councils and researchers.

What JTIS Provides

Clean, reproducible indicators:
Per-capita emissions, transport energy use, IMD mean rank, LAD × Year harmonised tables.

Composite Just Transition Index (JTI):
Three components (emissions, transport, deprivation) with national percentiles and ranks.

Place-based clusters:
Each LAD is assigned to a cluster describing typical transition contexts.

Interactive dashboard:
Executive Summary, Components, Trends, Comparisons, Clusters.

System Architecture

Raw data → Ingestion → Harmonisation → Indicators → Composite (JTI) → Profiles + Clusters → Dashboard

The pipeline is modular, offline, and fully reproducible.

Repository Structure

src/ingestion – raw dataset loaders
src/harmonisation – LAD/year alignment, validation
src/indicators – emissions, transport, deprivation indicators
src/composite – JTI and weighting
src/profiles – LAD profiles, cluster assignment
src/dashboard – Streamlit app
config/ – dataset and settings YAML
outputs/ – pipeline results (jti_scores.csv, lad_profile_summary.csv)
notebooks/ – exploration and validation

Usage

Create your environment and install dependencies

Place offline datasets as configured in config/datasets.yaml

Run the pipeline using run_pipeline.sh

Launch the dashboard using streamlit run src/dashboard/app.py

Outputs appear in the outputs/ folder.

Dashboard Overview

Overview – JTI score, rank, percentile, priority, cluster, narrative
Components – component scores for emissions, transport, deprivation
Trends – time-series for emissions, transport, IMD, JTI
Comparisons – national benchmarking, cluster peers, statistical peers
Clusters – cluster characteristics, distributions, narrative insights