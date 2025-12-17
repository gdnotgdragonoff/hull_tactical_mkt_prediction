import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import TARGET_COL, RANDOM_STATE
from .data_loading import time_based_split
from .paths import RESULTS_DIR, ARTIFACTS_DIR
from .utils import get_feature_cols, drop_dummy_cols, compute_feature_pair_corr


def _safe_numeric_cols(df, cols):
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def _keep_missing_flags(cols):
    return [c for c in cols if c.endswith("_missing")]


def _drop_high_missing(train_df, cols, thr=0.40):
    flags = set(_keep_missing_flags(cols))
    base_cols = [c for c in cols if c not in flags]

    miss_rate = train_df[base_cols].isnull().mean()
    dropped = miss_rate[miss_rate > thr].index.tolist()

    kept = [c for c in cols if c not in dropped]
    return kept, dropped, miss_rate


def _variance_filter(train_df, cols, q=0.25):
    var = train_df[cols].var(ddof=0)
    thr = float(var.quantile(q))
    dropped = var[var <= thr].index.tolist()
    kept = [c for c in cols if c not in dropped]
    return kept, dropped, var, thr


def _corr_filter_by_top_pairs(cols, var_series, top_pairs_df, corr_thr=0.95):
    candidates = cols.copy()
    dropped = set()

    df = top_pairs_df
    if "abs_corr" in df.columns:
        df = df[df["abs_corr"] >= corr_thr].sort_values("abs_corr", ascending=False)

    for _, row in df.iterrows():
        f1, f2 = row["feature_1"], row["feature_2"]
        if f1 not in candidates or f2 not in candidates:
            continue

        v1 = var_series.get(f1, np.nan)
        v2 = var_series.get(f2, np.nan)
        if np.isnan(v1) or np.isnan(v2):
            continue

        drop_col = f2 if v1 >= v2 else f1
        dropped.add(drop_col)
        candidates.remove(drop_col)

    return candidates, sorted(list(dropped))


def _train_lgbm_gain_importance(train_df, val_df, feature_cols):
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]

    tr = lgb.Dataset(X_train, y_train)
    va = lgb.Dataset(X_val, y_val, reference=tr)

    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": RANDOM_STATE,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        tr,
        valid_sets=[tr, va],
        valid_names=["train", "valid"],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    gain = model.feature_importance(importance_type="gain")
    imp = (
        pd.DataFrame({"feature": feature_cols, "gain": gain})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    return imp


def _select_by_cum_gain(importance_df, top_ratio=0.90):
    total = float(importance_df["gain"].sum())
    if total <= 0:
        return importance_df["feature"].tolist()

    cum = importance_df["gain"].cumsum()
    selected = importance_df.loc[cum <= top_ratio * total, "feature"].tolist()
    if not selected:
        selected = importance_df["feature"].head(1).tolist()
    return selected


def run_feature_selection(
    full_cleaned,
    high_missing_thr=0.40,
    var_quantile=0.25,
    corr_top_n=2000,
    corr_threshold=0.95,
    lgb_top_gain_ratio=0.90,
    save_prefix="feature_selection",
):
    """
    Expected by notebooks/scripts:
      - input: full_cleaned DataFrame (sorted by date_id preferred)
      - output: dict with selected_features (list) and importance_df (DataFrame)
      - side effects:
          results/correlation_stats_trainset.xlsx
          artifacts/{save_prefix}_importance_gain.csv
          artifacts/{save_prefix}_selected_features.json
    """

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_set, val_set, test_set = time_based_split(full_cleaned, 0.7, 0.2, sort_by="date_id")

    feat_cols = get_feature_cols(full_cleaned)
    feat_cols = _safe_numeric_cols(train_set, feat_cols)
    feat_cols = drop_dummy_cols(train_set, feat_cols)

    feat_cols, dropped_high_na, miss_rate = _drop_high_missing(train_set, feat_cols, thr=high_missing_thr)

    feat_cols, dropped_low_var, var_series, var_thr = _variance_filter(train_set, feat_cols, q=var_quantile)

    corr_stats = compute_feature_pair_corr(
        train_set,
        feature_cols=feat_cols,
        top_n=corr_top_n,
        method="pearson",
        excel_name="correlation_stats_trainset.xlsx",
    )
    top_pairs_df = corr_stats["top_pairs_df"]

    feat_cols, dropped_by_corr = _corr_filter_by_top_pairs(
        feat_cols,
        var_series=var_series,
        top_pairs_df=top_pairs_df,
        corr_thr=corr_threshold,
    )

    importance_df = _train_lgbm_gain_importance(train_set, val_set, feat_cols)
    selected = _select_by_cum_gain(importance_df, top_ratio=lgb_top_gain_ratio)

    imp_path = ARTIFACTS_DIR / f"{save_prefix}_importance_gain.csv"
    importance_df.to_csv(imp_path, index=False)

    out = {
        "raw_feature_count": int(len(get_feature_cols(full_cleaned))),
        "after_drop_dummy": int(len(drop_dummy_cols(train_set, _safe_numeric_cols(train_set, get_feature_cols(full_cleaned))))),
        "high_missing_thr": float(high_missing_thr),
        "dropped_high_missing": dropped_high_na,
        "var_quantile": float(var_quantile),
        "var_threshold": float(var_thr),
        "dropped_low_variance": dropped_low_var,
        "corr_threshold": float(corr_threshold),
        "dropped_by_corr": dropped_by_corr,
        "lgb_top_gain_ratio": float(lgb_top_gain_ratio),
        "selected_features": selected,
        "paths": {
            "correlation_xlsx": str(RESULTS_DIR / "correlation_stats_trainset.xlsx"),
            "importance_csv": str(imp_path),
        },
    }

    sel_path = ARTIFACTS_DIR / f"{save_prefix}_selected_features.json"
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return {
        "selected_features": selected,
        "importance_df": importance_df,
        "metadata": out,
        "train_shape": tuple(train_set.shape),
        "val_shape": tuple(val_set.shape),
        "test_shape": tuple(test_set.shape),
    }
