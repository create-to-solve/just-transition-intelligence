"""
ScoutAgent

Purpose
-------
Pre-flight checker for the JTIS pipeline.

This agent:
- Reads the dataset registry from config/datasets.yaml
- Resolves each dataset path relative to the repo root
- Checks:
    * file existence
    * basic readability (sample rows)
    * optional schema consistency (if validation schemas exist)
    * heuristic presence of LAD and year columns
- Writes a JSON diagnostics report to outputs/diagnostics/scout_report.json
- Prints a concise summary to stdout

This module does NOT modify any data or run the pipeline.
It is safe to run independently before any processing.
"""

from __future__ import annotations

import json
import datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import yaml

from src.utils.logger import get_logger



# ----------------------------------------------------------------------
# Paths and basic configuration
# ----------------------------------------------------------------------

# scout_agent.py is expected to live in src/agents/
# parents[0] = agents, parents[1] = src, parents[2] = repo root
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"

DATASETS_REGISTRY_PATH = CONFIG_DIR / "datasets.yaml"
VALIDATION_SCHEMAS_DIR = CONFIG_DIR / "validation_schemas"


logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass
class DatasetCheckResult:
    dataset_key: str
    name: str
    path: str
    exists: bool
    readable: bool
    n_rows: Optional[int]
    columns: List[str]
    schema_checked: bool
    schema_ok: Optional[bool]
    missing_columns: List[str]
    extra_columns: List[str]
    lad_column_guess: Optional[str]
    year_column_guess: Optional[str]
    errors: List[str]


@dataclass
class ScoutReport:
    timestamp_utc: str
    repo_root: str
    datasets_registry_path: str
    all_ok: bool
    datasets: List[DatasetCheckResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "repo_root": self.repo_root,
            "datasets_registry_path": self.datasets_registry_path,
            "all_ok": self.all_ok,
            "datasets": [asdict(d) for d in self.datasets],
        }

# ----------------------------------------------------------------------
# Schema loading utilities
# ----------------------------------------------------------------------

def load_validation_schema(dataset_key: str) -> Optional[Dict[str, Any]]:
    """
    Load and parse the schema YAML for a dataset.
    Returns None if the schema file does not exist.
    """
    schema_path = VALIDATION_SCHEMAS_DIR / f"{dataset_key}.yaml"
    if not schema_path.exists():
        return None
    try:
        with schema_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load YAML schema for %s: %s", dataset_key, exc)
        return None

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def _load_datasets_registry(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load the dataset registry from config/datasets.yaml.

    Expected structure (example):

    datasets:
      desnz_ghg:
        path: data/raw/desnz_ghg_emissions.csv
        description: DESNZ LA GHG emissions
      dft_fuel:
        path: data/raw/dft_fuel_consumption.xlsx
      ons_population:
        path: data/raw/ons_population.xlsx
      imd_2019:
        path: data/raw/imd_2019.xlsx

    If the file has a different structure, we fall back to treating the top-level
    keys as dataset keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset registry not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    if "datasets" in doc and isinstance(doc["datasets"], dict):
        return doc["datasets"]

    # Fallback: treat top-level mapping as datasets
    if isinstance(doc, dict):
        return doc

    raise ValueError(
        f"Unexpected structure in {path}. Expected 'datasets:' mapping or top-level mapping."
    )


def _load_validation_schema(dataset_key: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to load an optional validation schema for the dataset.

    Expected location:
        config/validation_schemas/<dataset_key>.yaml

    Expected minimal structure (example):

        expected_columns:
          - LAD Code
          - LAD Name
          - Year
          - Total emissions (kt CO2e)

    If the file doesn't exist or is invalid, returns None and schema
    validation is skipped for that dataset.
    """
    schema_path = VALIDATION_SCHEMAS_DIR / f"{dataset_key}.yaml"
    if not schema_path.exists():
        return None

    try:
        with schema_path.open("r", encoding="utf-8") as f:
            schema = yaml.safe_load(f) or {}
        if not isinstance(schema, dict):
            logger.warning("Schema for %s is not a dict; ignoring.", dataset_key)
            return None
        return schema
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load schema for %s: %s", dataset_key, exc)
        return None


def read_raw_with_schema(path: Path, schema: Dict[str, Any]) -> pd.DataFrame:
    """
    Load the raw dataset using header/skiprows/sheet_name rules specified
    in the validation schema.
    """
    read_kwargs = {}

    # Optional sheet selection (for Excel files)
    if "sheet_name" in schema:
        read_kwargs["sheet_name"] = schema["sheet_name"]

    # Optional row skipping
    if "skiprows" in schema:
        read_kwargs["skiprows"] = schema["skiprows"]

    # Header row specification
    if "expected_header_row" in schema:
        read_kwargs["header"] = schema["expected_header_row"]

    # Try reading
    suffix = path.suffix.lower()
    if suffix in [".xls", ".xlsx"]:
        return pd.read_excel(path, **read_kwargs)
    else:
        return pd.read_csv(path, **read_kwargs)

def validate_required_columns(df: pd.DataFrame, schema: Dict[str, Any]) -> (bool, List[str], List[str]):
    """
    Validate required columns defined under:
        required_columns:
            lad_code:
                any_of: [..]
            year:
                any_of: [..]
    """
    missing = []
    extra = []
    ok = True

    required = schema.get("required_columns", {})
    for logical_name, rule in required.items():
        allowed = rule.get("any_of", [])
        if not any(col in df.columns for col in allowed):
            missing.append(f"{logical_name}: {allowed}")
            ok = False

    return ok, missing, extra

def validate_wide_years(df: pd.DataFrame, schema: Dict[str, Any]) -> (bool, List[str]):
    """
    Validate presence of wide-format year columns, e.g.
    population_2002 ... population_2024
    """
    wy = schema.get("wide_years")
    if not wy:
        return True, []

    prefix = wy["prefix"]
    start = wy["allowed_year_range"]["start"]
    end = wy["allowed_year_range"]["end"]

    missing = []
    for year in range(start, end + 1):
        col = f"{prefix}{year}"
        if col not in df.columns:
            missing.append(col)

    ok = len(missing) == 0
    return ok, missing

def validate_numeric_columns(df: pd.DataFrame, schema: Dict[str, Any]) -> (bool, List[str]):
    rules = schema.get("column_rules", {})
    numeric_cols = rules.get("numeric_columns", [])

    non_numeric = []
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric.append(col)

    return len(non_numeric) == 0, non_numeric

def validate_row_count(df: pd.DataFrame, schema: Dict[str, Any]) -> (bool, int, int):
    expected = schema.get("row_rules", {}).get("min_rows")
    if not expected:
        return True, len(df), expected

    return len(df) >= expected, len(df), expected


def _guess_lad_column(columns: List[Any]) -> Optional[str]:
    """
    Heuristic to guess the LAD code column from a list of column names.
    Accepts integer or mixed-type column names by converting all to strings.
    """
    str_cols = [str(c) for c in columns]
    lower_cols = [c.lower() for c in str_cols]

    # Strong hints
    for target in ["lad22cd", "ladcd", "lad_code", "local authority code"]:
        for orig, low in zip(columns, lower_cols):
            if low == target:
                return str(orig)

    # Softer heuristics
    for orig, low in zip(columns, lower_cols):
        if "lad" in low and "code" in low:
            return str(orig)
        if "local authority" in low and ("code" in low or "cd" in low):
            return str(orig)

    return None


def _guess_year_column(columns: List[Any]) -> Optional[str]:
    """
    Heuristic to guess the year column; robust to integer column names.
    """
    str_cols = [str(c) for c in columns]
    lower_cols = [c.lower() for c in str_cols]

    for target in ["year", "yr", "calendar_year"]:
        for orig, low in zip(columns, lower_cols):
            if low == target:
                return str(orig)

    for orig, low in zip(columns, lower_cols):
        if "year" in low:
            return str(orig)

    return None



def _compare_schema(
    dataset_key: str,
    df: pd.DataFrame,
    schema: Dict[str, Any],
) -> tuple[bool, List[str], List[str]]:
    """
    Compare DataFrame columns with a schema definition.

    Currently supports:
      - schema["expected_columns"]: list of column names that MUST be present.

    Returns:
      (schema_ok, missing_columns, extra_columns)
    """
    expected = schema.get("expected_columns")
    if not expected:
        # No explicit expectations; treat as not checked.
        return False, [], []

    df_cols = list(df.columns)
    missing = [c for c in expected if c not in df_cols]
    # "Extra" is more subjective; we expose it but don't treat as fatal.
    extra = [c for c in df_cols if c not in expected]

    schema_ok = len(missing) == 0
    if not schema_ok:
        logger.warning(
            "Schema mismatch for %s: missing columns: %s",
            dataset_key,
            missing,
        )
    return True, missing, extra


# ----------------------------------------------------------------------
# ScoutAgent
# ----------------------------------------------------------------------


class ScoutAgent:
    """
    Pre-ingestion validation and diagnostics for JTIS.

    Usage
    -----
    From the repo root:

        python -m src.agents.scout_agent

    Or from another module:

        from src.agents.scout_agent import ScoutAgent

        agent = ScoutAgent()
        report = agent.run()
        agent.save_report(report)  # writes JSON to outputs/diagnostics

    This agent is read-only and safe to run at any time.
    """

    def __init__(
        self,
        datasets_registry_path: Path | None = None,
        diagnostics_dir: Path | None = None,
    ) -> None:
        self.datasets_registry_path = datasets_registry_path or DATASETS_REGISTRY_PATH
        self.diagnostics_dir = diagnostics_dir or DIAGNOSTICS_DIR

    # ------------------------- public API -------------------------

    def run(self) -> ScoutReport:
        """
        Run all dataset checks and build a ScoutReport.
        """
        logger.info("ScoutAgent starting. Registry: %s", self.datasets_registry_path)

        registry = _load_datasets_registry(self.datasets_registry_path)
        results: List[DatasetCheckResult] = []

        for key, meta in registry.items():
            result = self._check_single_dataset(key, meta)
            results.append(result)

        all_ok = all(r.exists and r.readable and (r.schema_ok is not False) for r in results)

        report = ScoutReport(
            timestamp_utc=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            repo_root=str(ROOT_DIR),
            datasets_registry_path=str(self.datasets_registry_path),
            all_ok=all_ok,
            datasets=results,
        )

        logger.info("ScoutAgent finished. all_ok=%s", all_ok)
        return report

    def save_report(self, report: ScoutReport, filename: str = "scout_report.json") -> Path:
        """
        Persist the report as JSON under outputs/diagnostics.
        """
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.diagnostics_dir / filename
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("ScoutAgent report written to %s", out_path)
        return out_path

    def run_and_save(self) -> ScoutReport:
        """
        Convenience wrapper: run checks, save report, print summary.
        """
        report = self.run()
        self.save_report(report)
        self._print_summary(report)
        return report

    # ------------------------- internals -------------------------

    def _check_single_dataset(self, key: str, meta: Dict[str, Any]) -> DatasetCheckResult:
        """
        Improved ingestion logic:
        - Reads loader/sheet/header_rows_to_skip from datasets.yaml
        - Uses that configuration to read the raw file
        - Then applies schema validation
        """
        errors = []
        name = meta.get("name") or meta.get("description") or key

        # -----------------------------
        # Resolve path
        # -----------------------------
        raw_path = meta.get("path") or meta.get("filepath") or meta.get("file")
        if raw_path is None:
            errors.append("No path found in dataset definition.")
            return DatasetCheckResult(
                dataset_key=key, name=name, path="<missing>",
                exists=False, readable=False, n_rows=None, columns=[],
                schema_checked=False, schema_ok=None,
                missing_columns=[], extra_columns=[],
                lad_column_guess=None, year_column_guess=None,
                errors=errors
            )

        path = (ROOT_DIR / raw_path).resolve()
        if not path.exists():
            errors.append(f"File does not exist: {path}")
            return DatasetCheckResult(
                dataset_key=key, name=name, path=str(path),
                exists=False, readable=False, n_rows=None, columns=[],
                schema_checked=False, schema_ok=None,
                missing_columns=[], extra_columns=[],
                lad_column_guess=None, year_column_guess=None,
                errors=errors
            )

        # -----------------------------
        # Ingestion config from datasets.yaml
        # -----------------------------
        loader = meta.get("loader", None)
        sheet = meta.get("sheet")
        sheets = meta.get("sheets")
        skiprows = meta.get("header_rows_to_skip")

        read_kwargs = {}
        if skiprows is not None:
            read_kwargs["skiprows"] = skiprows
            read_kwargs["header"] = 0

        if loader == "excel":
            if sheet:
                read_kwargs["sheet_name"] = sheet
            elif sheets:
                read_kwargs["sheet_name"] = sheets[0]

        # -----------------------------
        # Load schema (optional)
        # -----------------------------
        schema = load_validation_schema(key)
        schema_checked = schema is not None
        schema_ok = None

        # -----------------------------
        # Try reading the raw file
        # -----------------------------
        try:
            suffix = path.suffix.lower()
            if loader == "excel" or suffix in [".xls", ".xlsx"]:
                df = pd.read_excel(path, **read_kwargs)
            else:
                df = pd.read_csv(path, **read_kwargs)
            readable = True
        except Exception as exc:
            readable = False
            errors.append(f"Failed to read file: {exc}")

            return DatasetCheckResult(
                dataset_key=key, name=name, path=str(path),
                exists=True, readable=False, n_rows=None, columns=[],
                schema_checked=schema_checked, schema_ok=False,
                missing_columns=["<entire file unreadable>"], extra_columns=[],
                lad_column_guess=None, year_column_guess=None,
                errors=errors
            )

        # -----------------------------
        # Schema validation
        # -----------------------------
        missing_total = []
        if schema:
            ok_req, missing_req, _ = validate_required_columns(df, schema)
            missing_total.extend(missing_req)

            ok_wy, missing_wy = validate_wide_years(df, schema)
            missing_total.extend(missing_wy)

            ok_num, non_numeric = validate_numeric_columns(df, schema)
            if non_numeric:
                missing_total.append(f"Non-numeric: {non_numeric}")

            ok_rows, actual_rows, expected_rows = validate_row_count(df, schema)
            if not ok_rows:
                missing_total.append(f"Row count {actual_rows} < expected {expected_rows}")

            schema_ok = ok_req and ok_wy and ok_num and ok_rows

        # -----------------------------
        # LAD/year guess
        # -----------------------------
        lad_guess = _guess_lad_column(df.columns)
        year_guess = _guess_year_column(df.columns)

        # -----------------------------
        # Build result
        # -----------------------------
        return DatasetCheckResult(
            dataset_key=key, name=name, path=str(path),
            exists=True, readable=True,
            n_rows=len(df), columns=list(df.columns),
            schema_checked=schema_checked, schema_ok=schema_ok,
            missing_columns=missing_total, extra_columns=[],
            lad_column_guess=lad_guess, year_column_guess=year_guess,
            errors=errors
        )


    @staticmethod
    def _print_summary(report: ScoutReport) -> None:
        """
        Print a concise human-readable summary to stdout.
        """
        print("\n=== JTIS ScoutAgent Report ===")
        print(f"Timestamp (UTC): {report.timestamp_utc}")
        print(f"Repo root      : {report.repo_root}")
        print(f"Registry       : {report.datasets_registry_path}")
        print(f"All OK         : {report.all_ok}")
        print()
        for ds in report.datasets:
            status_parts = []
            status_parts.append("OK" if ds.exists and ds.readable else "BROKEN")
            if ds.schema_ok is False:
                status_parts.append("SCHEMA MISMATCH")
            status = " | ".join(status_parts)

            print(f"[{ds.dataset_key}] {ds.name}")
            print(f"  Path      : {ds.path}")
            print(f"  Status    : {status}")
            print(f"  Exists    : {ds.exists}")
            print(f"  Readable  : {ds.readable}")
            print(f"  Rows(sample): {ds.n_rows}")
            print(f"  LAD col   : {ds.lad_column_guess}")
            print(f"  Year col  : {ds.year_column_guess}")
            if ds.missing_columns:
                print(f"  Missing cols: {ds.missing_columns}")
            if ds.errors:
                print(f"  Errors      : {ds.errors}")
            print()
        print("Report saved to outputs/diagnostics/scout_report.json")
        print("==============================\n")


# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------


def main() -> None:
    """
    Entrypoint for `python -m src.agents.scout_agent`.
    """
    agent = ScoutAgent()
    agent.run_and_save()


if __name__ == "__main__":
    main()
