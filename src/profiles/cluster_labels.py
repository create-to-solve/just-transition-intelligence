"""
Profile Clustering Utilities (JTIS v2)

Extracted from v1 `lad_profiles.py` and refactored into a clean,
reusable module. No behavioural changes have been introduced.

Provides:
    - kmeans_cluster_profiles: compute k-means cluster IDs
    - map_cluster_label: translate cluster IDs into interpretable labels
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Feature set used in v1
PROFILE_CLUSTER_FEATURES = [
    "jti_last_percentile",
    "jti_delta",
    "component_deprivation_percentile",
    "component_emissions_percentile",
    "component_transport_percentile",
]


def kmeans_cluster_profiles(df: pd.DataFrame, k: int = 5, random_state: int = 42):
    """
    Cluster LADs based on their profile statistics.
    Behaviour identical to v1 (k=5, StandardScaler).

    Parameters
    ----------
    df : DataFrame
        Must contain PROFILE_CLUSTER_FEATURES.
    k : int
        Number of clusters.
    random_state : int

    Returns
    -------
    cluster_ids : np.ndarray of length len(df)
    mask : boolean mask (True = row used in clustering)
    """
    # Drop rows with missing required features
    mask = df[PROFILE_CLUSTER_FEATURES].notna().all(axis=1)
    valid = df.loc[mask, PROFILE_CLUSTER_FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(valid)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    cluster_ids = kmeans.fit_predict(X_scaled)

    return cluster_ids, mask


def map_cluster_label(cid):
    """
    Map cluster ID to descriptive label.
    Identical to v1.
    """
    if cid is None or (isinstance(cid, float) and np.isnan(cid)):
        return "Unassigned"

    cid = int(cid)

    return {
        0: "Transition Leaders",
        1: "Emerging Improvers",
        2: "Stable Mid-Performers",
        3: "High-Deprivation Barriers",
        4: "Transition Laggards",
    }.get(cid, f"Cluster {cid}")
