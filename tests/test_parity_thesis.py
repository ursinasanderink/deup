"""P11: thesis finance g(x) parity — synthetic smoke + optional live gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deup.domains.finance_walkforward import THESIS_G_PARAMS, walkforward_g_on_enriched

ROOT = Path(__file__).resolve().parents[1]
PARITY_SCRIPT = ROOT / "scripts" / "parity_thesis_finance.py"
THESIS_DATA = Path(
    "/Users/ursinasanderink/Downloads/AI Stock Forecast/evaluation_outputs"
    "/chapter13_v3/enriched_residuals_tabular_lgb.parquet"
)
FROZEN_G = Path(
    "/Users/ursinasanderink/Downloads/AI Stock Forecast/evaluation_outputs"
    "/chapter13_v3/g_predictions_rank.parquet"
)


def _synthetic_enriched(n_folds: int = 25, n_assets: int = 40, seed: int = 0) -> pd.DataFrame:
    """Minimal enriched panel matching thesis schema."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for fi in range(1, n_folds + 1):
        fold = f"fold_{fi:02d}"
        for _ in range(n_assets):
            score = float(rng.normal())
            rows.append(
                {
                    "as_of_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=fi * 30),
                    "ticker": "AAA",
                    "stable_id": 1,
                    "horizon": 20,
                    "fold_id": fold,
                    "score": score,
                    "rank_loss": abs(float(rng.normal(scale=0.2))),
                    "vol_20d": float(rng.uniform(0.1, 0.5)),
                    "vol_60d": float(rng.uniform(0.1, 0.5)),
                    "mom_1m": float(rng.normal(scale=0.1)),
                    "adv_20d": float(rng.uniform(1e6, 5e6)),
                    "vix_percentile_252d": float(rng.uniform(0, 1)),
                    "market_regime_enc": float(rng.choice([-1.0, 0.0, 1.0])),
                    "market_vol_21d": float(rng.uniform(0.1, 0.3)),
                    "market_return_21d": float(rng.normal(scale=0.05)),
                }
            )
    return pd.DataFrame(rows)


def test_walkforward_g_synthetic_runs() -> None:
    df = _synthetic_enriched()
    preds, diag = walkforward_g_on_enriched(df, min_train_folds=5, horizons=[20])
    assert not preds.empty
    assert "g_pred" in preds.columns
    assert np.all(preds["g_pred"] >= 0.0)
    hz_diag = diag.get("20") or diag.get(20)
    assert hz_diag is not None
    assert hz_diag["spearman_rho"] == hz_diag["spearman_rho"]  # finite


def test_thesis_g_params_match_documentation() -> None:
    assert THESIS_G_PARAMS["n_estimators"] == 50
    assert THESIS_G_PARAMS["random_state"] == 42


@pytest.mark.integration
@pytest.mark.skipif(not THESIS_DATA.exists(), reason="thesis enriched parquet not on disk")
def test_live_parity_script_passes() -> None:
    """Full parity gate when thesis artifacts are available locally."""
    pytest.importorskip("lightgbm")
    out = ROOT / "parity_results_test.json"
    cmd = [
        sys.executable,
        str(PARITY_SCRIPT),
        "--enriched",
        str(THESIS_DATA),
        "--frozen-g",
        str(FROZEN_G),
        "--thesis-root",
        str(THESIS_DATA.parents[2]),
        "--horizon",
        "20",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out.read_text())
    for row in report["comparisons"]:
        if row.get("n_merged", 0) > 0:
            assert row["max_abs_diff"] == 0.0
