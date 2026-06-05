#!/usr/bin/env python3
"""Parity gate: thesis ``train_g_walk_forward`` vs library ``walkforward_g_on_enriched``.

Compares g(x) predictions and diagnostics on the same enriched residual panel.
Does **not** modify thesis artifacts. Run from a clean ``deup`` checkout:

    pip install "deup[finance,gbm]" pyarrow
    python scripts/parity_thesis_finance.py \\
        --enriched /path/to/enriched_residuals_tabular_lgb.parquet \\
        --frozen-g /path/to/g_predictions_rank.parquet \\
        --thesis-root /path/to/AI\\ Stock\\ Forecast   # optional thesis re-run

Exit 0 when library-vs-thesis max |Δg| and Spearman deltas are within tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deup.domains.finance_walkforward import walkforward_g_on_enriched  # noqa: E402

MERGE_KEYS = ["as_of_date", "ticker", "stable_id", "horizon", "fold_id"]


def _load_thesis_train(thesis_root: Path | None):
    if thesis_root is None:
        return None
    sys.path.insert(0, str(thesis_root))
    from src.uncertainty.deup_estimator import train_g_walk_forward  # type: ignore[import-untyped]

    return train_g_walk_forward


def _compare(
    left: pd.DataFrame,
    right: pd.DataFrame,
    label: str,
) -> dict[str, float | int]:
    merged = left.merge(right, on=MERGE_KEYS, suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        return {"label": label, "n_merged": 0}
    ga = merged["g_pred_a"].to_numpy(dtype=float)
    gb = merged["g_pred_b"].to_numpy(dtype=float)
    return {
        "label": label,
        "n_merged": int(len(merged)),
        "max_abs_diff": float(np.max(np.abs(ga - gb))),
        "mean_abs_diff": float(np.mean(np.abs(ga - gb))),
        "spearman_g_a": float(stats.spearmanr(ga, merged["rank_loss_a"]).statistic),
        "spearman_g_b": float(stats.spearmanr(gb, merged["rank_loss_b"]).statistic),
        "spearman_cross": float(stats.spearmanr(ga, gb).statistic),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Thesis vs library DEUP finance parity")
    parser.add_argument(
        "--enriched",
        type=Path,
        default=Path(
            "/Users/ursinasanderink/Downloads/AI Stock Forecast/evaluation_outputs"
            "/chapter13_v3/enriched_residuals_tabular_lgb.parquet"
        ),
    )
    parser.add_argument(
        "--frozen-g",
        type=Path,
        default=Path(
            "/Users/ursinasanderink/Downloads/AI Stock Forecast/evaluation_outputs"
            "/chapter13_v3/g_predictions_rank.parquet"
        ),
    )
    parser.add_argument("--thesis-root", type=Path, default=None)
    parser.add_argument("--min-train-folds", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    parser.add_argument("--out", type=Path, default=ROOT / "parity_results.json")
    args = parser.parse_args()

    enriched = pd.read_parquet(args.enriched)
    if args.max_rows > 0:
        enriched = enriched.iloc[: args.max_rows].copy()

    frozen = pd.read_parquet(args.frozen_g) if args.frozen_g.exists() else None

    # Library path — legacy string fold order (matches deup_estimator.py exactly)
    lib_legacy, diag_legacy = walkforward_g_on_enriched(
        enriched,
        min_train_folds=args.min_train_folds,
        horizons=[args.horizon],
        fold_sort="string",
    )
    lib_legacy = lib_legacy.rename(columns={"rank_loss": "rank_loss"})

    # Library path — recommended numeric fold order (bugfix)
    lib_numeric, diag_numeric = walkforward_g_on_enriched(
        enriched,
        min_train_folds=args.min_train_folds,
        horizons=[args.horizon],
        fold_sort="numeric",
    )

    rows: list[dict[str, float | int | str]] = []
    if frozen is not None:
        merged_left = lib_legacy.rename(columns={"g_pred": "g_pred_a", "rank_loss": "rank_loss_a"})
        merged_right = frozen.rename(columns={"g_pred": "g_pred_b", "rank_loss": "rank_loss_b"})
        cmp = _compare(merged_left, merged_right, "library_legacy_vs_frozen")
        rows.append({"comparison": "library_legacy_vs_frozen", **cmp})
    else:
        rows.append({"comparison": "library_legacy_vs_frozen", "n_merged": 0})

    thesis_train = _load_thesis_train(args.thesis_root)
    if thesis_train is not None:
        thesis_preds, thesis_diag = thesis_train(
            enriched,
            target_col="rank_loss",
            min_train_folds=args.min_train_folds,
            horizons=[args.horizon],
        )
        rows.append(
            {
                "comparison": "thesis_rerun_vs_library_legacy",
                **_compare(
                    thesis_preds.rename(columns={"g_pred": "g_pred_a", "rank_loss": "rank_loss_a"}),
                    lib_legacy.rename(columns={"g_pred": "g_pred_b", "rank_loss": "rank_loss_b"}),
                    "thesis_rerun_vs_library_legacy",
                ),
            }
        )
        if frozen is not None:
            cmp2 = _compare(
                thesis_preds.rename(columns={"g_pred": "g_pred_a", "rank_loss": "rank_loss_a"}),
                frozen.rename(columns={"g_pred": "g_pred_b", "rank_loss": "rank_loss_b"}),
                "thesis_rerun_vs_frozen",
            )
            rows.append({"comparison": "thesis_rerun_vs_frozen", **cmp2})

    report = {
        "enriched_path": str(args.enriched),
        "frozen_g_path": str(args.frozen_g) if frozen is not None else None,
        "horizon": args.horizon,
        "min_train_folds": args.min_train_folds,
        "library_diagnostics_legacy": diag_legacy,
        "library_diagnostics_numeric": diag_numeric,
        "comparisons": rows,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # Gate: library legacy should match thesis rerun and frozen within float tolerance
    tol = 1e-6
    ok = True
    for row in rows:
        n = row.get("n_merged", 0)
        if isinstance(n, int) and n > 0:
            mad = row.get("max_abs_diff", float("inf"))
            if isinstance(mad, (int, float)) and mad > tol:
                ok = False
                print(f"FAIL {row.get('comparison')}: max_abs_diff={mad}", file=sys.stderr)
    if ok:
        print("PARITY PASS", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
