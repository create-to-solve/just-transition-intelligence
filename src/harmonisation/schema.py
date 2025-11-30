"""
Canonical JTIS Schema Definitions

This module defines the core field names and schema used throughout
the harmonisation layer. The aim is to ensure that all ingested
datasets are transformed into a consistent LAD–year long-form shape.

No transformation logic is implemented here — only constants.
"""

# Canonical unit for analysis
UNIT_GEO = "lad"
UNIT_TIME = "year"

# Expected canonical fields after harmonisation
CANONICAL_FIELDS = [
    "lad_code",
    "lad_name",
    "year",
]
