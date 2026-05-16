"""Load and clean the biomaterial degradation spreadsheet."""

from pathlib import Path

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def load_training_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the Excel file and return only the columns needed for modeling."""
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find the data file: {data_path}")

    data = pd.read_excel(data_path)

    # Some spreadsheets have extra empty columns. Remove them before checking
    # whether all required columns exist.
    data.columns = [str(column).strip() for column in data.columns]
    data = data.dropna(axis=1, how="all")
    data = data.loc[:, ~data.columns.str.startswith("Unnamed")]

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"The spreadsheet is missing these columns: {missing_text}")

    cleaned = data[required_columns].copy()

    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].replace("", pd.NA)

    # Drop rows that cannot be used for supervised learning.
    cleaned = cleaned.dropna(subset=required_columns).reset_index(drop=True)

    # Convert string extension arrays back to regular Python strings so
    # scikit-learn can process them without surprises.
    for column in CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].astype(str)

    return cleaned


def get_category_options(data: pd.DataFrame, column: str) -> list[str]:
    """Return sorted choices for a categorical column."""
    return sorted(data[column].dropna().astype(str).unique().tolist())
