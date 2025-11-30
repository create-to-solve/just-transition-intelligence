"""
Validation and QA for Harmonised Data

These functions perform basic consistency checks:
    - Are all LAD codes valid?
    - Are all expected years present?
    - Are there duplicate LAD–year rows?
    - Are numeric fields within plausible ranges?

At this phase, only placeholders exist. Real validation logic will be
ported later.
"""

def validate_harmonised(df):
    """
    Validate harmonised LAD–year dataset.

    TODO:
        - Implement checks from old repo.
        - Add schema validation.
        - Validate year ranges and LAD counts.
    """
    raise NotImplementedError("Harmonised data validation not implemented.")
