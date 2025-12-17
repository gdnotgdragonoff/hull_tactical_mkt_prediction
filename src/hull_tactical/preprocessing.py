import pandas as pd
import numpy as np

from .data_loading import time_based_split
from .paths import RESULTS_DIR
from .utils import get_feature_cols


def _replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def _add_missing_flags(df: pd.DataFrame, feature_cols: list[str], na_thr: float = 0.40):
    """
    Create *_missing indicator columns for features whose missing rate > na_thr.
    Missing rate is computed on the full cleaned DataFrame (not just train split),
    because the goal is to preserve information about missingness patterns.
    """
    df = df.copy()

    miss_rate = df[feature_cols].isnull().mean()
    high_na_cols = miss_rate[miss_rate > na_thr].index.tolist()

    for c in high_na_cols:
        df[f"{c}_missing"] = df[c].isnull().astype(int)

    return df, high_na_cols, miss_rate


def build_cleaned_data(
    train_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    sort_by: str = "date_id",
    high_na_thr: float = 0.40,
    save_excel: bool = True,
    excel_name: str = "cleaned_train.xlsx",
):
    """
    Preprocessing entrypoint used by scripts and notebooks.

    Returns
    -------
    full_cleaned : pd.DataFrame
        Cleaned full dataset with *_missing flags added.
    train_clean, val_clean, test_clean : pd.DataFrame
        Time-based splits from full_cleaned.
    high_na_cols : list[str]
        Feature columns whose missing rate > high_na_thr and thus got *_missing flags.
    """
    if sort_by not in train_df.columns:
        raise ValueError(f"Expected '{sort_by}' column for time sorting, but not found.")

    df = train_df.sort_values(sort_by).reset_index(drop=True).copy()

    # Basic numeric sanitization
    df = _replace_inf_with_nan(df)

    # Identify candidate feature columns (excludes meta/finance cols via utils.get_feature_cols)
    feature_cols = get_feature_cols(df)

    # Add missingness flags for high-NA features
    full_cleaned, high_na_cols, miss_rate = _add_missing_flags(
        df, feature_cols=feature_cols, na_thr=high_na_thr
    )

    # Time-based split
    train_clean, val_clean, test_clean = time_based_split(
        full_cleaned, train_ratio=train_ratio, val_ratio=val_ratio, sort_by=sort_by
    )

    # Optional export for inspection / reproducibility
    if save_excel:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / excel_name

        with pd.ExcelWriter(out_path) as writer:
            full_cleaned.to_excel(writer, sheet_name="full_cleaned_with_flags", index=False)
            train_clean.to_excel(writer, sheet_name="train_cleaned", index=False)
            val_clean.to_excel(writer, sheet_name="val_cleaned", index=False)
            test_clean.to_excel(writer, sheet_name="test_cleaned", index=False)

            pd.DataFrame(
                {
                    "feature": miss_rate.index,
                    "missing_rate": miss_rate.values,
                    "is_high_na": miss_rate.index.isin(high_na_cols),
                }
            ).sort_values("missing_rate", ascending=False).to_excel(
                writer, sheet_name="missing_summary", index=False
            )

        print(f"Saved cleaned splits + missing summary to: {out_path}")

    return full_cleaned, train_clean, val_clean, test_clean, high_na_cols
