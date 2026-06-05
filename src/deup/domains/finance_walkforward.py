"""Walk-forward g(x) on pre-enriched finance residuals (thesis migration path).

Thesis Chapter 13 trains ``g`` on *already computed* walk-forward rank losses from the
primary model — it does not re-run the full DEUP OOF loop on raw features. This module
is the library re-expression of ``src/uncertainty/deup_estimator.train_g_walk_forward``.
"""

from __future__ import annotations

from typing import Any, Literal

from deup.core.error_estimator import ErrorEstimator
from deup.domains.finance import FINANCE_G_FEATURES, enrich_panel

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

# Thesis LightGBM hyperparameters (deup_estimator.G_PARAMS).
THESIS_G_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 50,
    "max_depth": 3,
    "num_leaves": 8,
    "min_child_samples": 50,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

FoldSort = Literal["numeric", "string"]


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "walkforward_g_on_enriched requires pandas. Install with: pip install deup[finance]"
        )
    return pd


def available_finance_features(df: Any, *, min_coverage: float = 0.5) -> list[str]:
    """Subset of :data:`FINANCE_G_FEATURES` present with sufficient non-null rate."""
    return [
        f for f in FINANCE_G_FEATURES if f in df.columns and df[f].notna().mean() > min_coverage
    ]


def _sort_folds(folds: list[str], method: FoldSort) -> list[str]:
    if method == "numeric":
        return sorted(folds, key=lambda x: int(str(x).split("_")[1]))
    return sorted(folds)


def walkforward_g_on_enriched(
    enriched: Any,
    *,
    target_col: str = "rank_loss",
    min_train_folds: int = 20,
    horizons: list[int] | None = None,
    date_col: str = "as_of_date",
    fold_sort: FoldSort = "numeric",
    g_params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Thesis-equivalent walk-forward ``g(x)`` on enriched residual panels.

    Fits :class:`~deup.core.error_estimator.ErrorEstimator` with LightGBM on each
    expanding fold window — the direct migration of ``train_g_walk_forward``.

    Parameters
    ----------
    enriched:
        Panel with ``fold_id``, ``horizon``, g-features, and ``target_col``.
    fold_sort:
        ``"numeric"`` (recommended) sorts ``fold_02 < fold_10 < fold_100``;
        ``"string"`` reproduces legacy thesis lexicographic order for frozen parity.
    """
    _require_pandas()
    from lightgbm import LGBMRegressor
    from scipy import stats

    df = enrich_panel(enriched, date_col=date_col)
    feats = available_finance_features(df)
    if not feats:
        raise ValueError("No g(x) features available after filtering")

    if horizons is None:
        horizons = sorted(int(h) for h in df["horizon"].unique())

    params = dict(THESIS_G_PARAMS if g_params is None else g_params)
    all_folds = _sort_folds([str(f) for f in df["fold_id"].unique()], fold_sort)
    predict_folds = all_folds[min_train_folds:]

    results: list[Any] = []
    diagnostics: dict[str, Any] = {"features": feats, "fold_sort": fold_sort}

    for hz in horizons:
        hz_data = df[df["horizon"] == hz]
        hz_preds: list[Any] = []

        for fold_idx, fold_id in enumerate(predict_folds):
            train_folds = set(all_folds[: min_train_folds + fold_idx])
            train = hz_data[hz_data["fold_id"].isin(train_folds)]
            test = hz_data[hz_data["fold_id"] == fold_id]
            if test.empty:
                continue

            x_train = train[feats].fillna(0.0).to_numpy(dtype=float)
            y_train = train[target_col].fillna(0.0).to_numpy(dtype=float)
            x_test = test[feats].fillna(0.0).to_numpy(dtype=float)

            est = ErrorEstimator(
                model=LGBMRegressor(**params),
                target_transform="none",
                clip_negative=True,
            )
            est.fit(x_train, y_train)
            preds = est.predict(x_test)

            out = test[[date_col, "ticker", "stable_id", "horizon", "fold_id", target_col]].copy()
            out["g_pred"] = preds
            hz_preds.append(out)

        if hz_preds:
            hz_df = pd.concat(hz_preds, ignore_index=True)
            rho = float(stats.spearmanr(hz_df["g_pred"], hz_df[target_col]).statistic)
            diagnostics[str(hz)] = {
                "n_rows": len(hz_df),
                "n_folds": int(hz_df["fold_id"].nunique()),
                "spearman_rho": rho,
            }
            results.append(hz_df)

    predictions = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return predictions, diagnostics
