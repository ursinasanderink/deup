# Migrating from thesis `deup_estimator.py` to `deup`

This guide maps the **Chapter 13 finance DEUP path** from the thesis codebase
(`src/uncertainty/deup_estimator.py`) to the installable library. Frozen thesis
results were **not** modified; parity was verified on held-out walk-forward folds.

## Two migration paths

| Path | When to use | Library entry point |
|---|---|---|
| **A. g-only on enriched residuals** | You already have walk-forward `score` + `rank_loss` (thesis Step 0 done) | `walkforward_g_on_enriched` |
| **B. End-to-end greenfield** | New project; train primary ranker + g together | `CrossSectionalDEUP` |

Path **A** is a drop-in replacement for `train_g_walk_forward`. Path **B** runs the
full DEUP meta-algorithm (OOF error collection + `ErrorEstimator` + rank
residualization + `HealthIndex`) and is the recommended default for new work.

---

## Before (thesis)

```python
from src.uncertainty.deup_estimator import train_g_walk_forward, prepare_g_features

enriched = prepare_g_features(enriched_residuals_df)  # adds abs_score, cross_sectional_rank, ...
g_preds, diagnostics = train_g_walk_forward(
    enriched,
    target_col="rank_loss",
    min_train_folds=20,
    horizons=[20, 60, 90],
)
# g_preds columns: as_of_date, ticker, stable_id, horizon, fold_id, rank_loss, g_pred
# diagnostics[20]["spearman_rho"]  →  ρ(g, rank_loss)
```

Internally this is LightGBM walk-forward on `G_FEATURES` with `G_PARAMS`, clipping
`g ≥ 0`.

---

## After (library) — Path A: direct parity

```python
from deup.domains.finance_walkforward import walkforward_g_on_enriched

g_preds, diagnostics = walkforward_g_on_enriched(
    enriched_residuals_df,
    target_col="rank_loss",
    min_train_folds=20,
    horizons=[20, 60, 90],
    fold_sort="numeric",   # recommended; see “Known diffs” below
)
```

Uses `ErrorEstimator` + `LGBMRegressor(**THESIS_G_PARAMS)` + `target_transform="none"`.
Feature selection matches thesis `_available_features` / `G_FEATURES`.

### Re-run parity locally

```bash
pip install "deup[finance,gbm]" pyarrow
python scripts/parity_thesis_finance.py \
  --enriched /path/to/enriched_residuals_tabular_lgb.parquet \
  --frozen-g /path/to/g_predictions_rank.parquet \
  --thesis-root /path/to/AI-Stock-Forecast   # optional thesis re-run
```

---

## After (library) — Path B: end-to-end preset

```python
from deup.domains.finance import CrossSectionalDEUP

model = CrossSectionalDEUP(horizon=20, cv=5, embargo=1).fit(panel_df)
model.calibrate(cal_panel, alpha=0.1)
pred, epistemic = model.predict(test_panel, return_uncertainty=True)
health = model.health_report(test_panel)   # composite context gate (Finding 2)
```

Adds rank-geometry residualization (Finding 3), conformal intervals, and
`HealthIndex` — not present in the original thesis `train_g_walk_forward`.

---

## Parity table (Chapter 13 v3, H=20)

Data: `evaluation_outputs/chapter13_v3/enriched_residuals_tabular_lgb.parquet`
(611,399 rows, 119 folds) vs frozen `g_predictions_rank.parquet`.
`min_train_folds=20`, legacy string fold order.

| Comparison | Rows merged | max \|Δg\| | mean \|Δg\| | ρ(g, rank_loss) |
|---|---:|---:|---:|---:|
| Library (legacy) vs frozen | 182,046 | **0.0** | **0.0** | 0.1885 |
| Thesis re-run vs library | 182,046 | **0.0** | **0.0** | 0.1885 |
| Thesis re-run vs frozen | 182,046 | **0.0** | **0.0** | 0.1885 |

**Verdict:** exact numerical parity on all 182,046 prediction rows (H=20, folds
`fold_11`…`fold_99` under legacy ordering).

With **numeric fold sort** (recommended bugfix), H=20 ρ(g, rank_loss) = **0.1917**
(+0.003 vs legacy) on 186,874 rows including folds `fold_21`…`fold_119`.

---

## Known diffs (refactor, not regression)

1. **Fold ordering (`fold_sort`)** — Thesis `deup_estimator.py` uses lexicographic
   `sorted(fold_id)` so `fold_100` sorts before `fold_11`. The library defaults to
   **numeric** order (`fold_02 < fold_10 < fold_100`). Pass `fold_sort="string"` for
   byte-identical reproduction of frozen thesis artifacts.

2. **Target transform** — Thesis trains on raw `rank_loss`. Library
   `ErrorEstimator` defaults to `log`; the migration helper sets
   `target_transform="none"` explicitly.

3. **End-to-end vs g-only** — `CrossSectionalDEUP` re-runs OOF primary-model training;
   it is not expected to match pre-enriched `g_pred` unless you configure the same
   primary model and splitter.

4. **Rank residualization & HealthIndex** — Library additions (Findings 2–3); apply
   on top of Path A or use Path B.

---

## Feature mapping

| Thesis `G_FEATURES` | Library |
|---|---|
| `prepare_g_features` | `enrich_panel(..., date_col="as_of_date")` |
| `_available_features` | `available_finance_features` |
| `G_PARAMS` | `THESIS_G_PARAMS` in `finance_walkforward` |
| `train_g_walk_forward` | `walkforward_g_on_enriched` |

Credit/breadth columns (`credit_ratio`, `breadth_ratio`, `downside_rv_share`) are in
`FINANCE_G_FEATURES` and used when present in the panel (not in v3 enriched snapshot).

---

## What we did **not** migrate here

- Primary signal model walk-forward (Chapter 7 `eval_rows`) — unchanged in thesis repo.
- Frozen JSON / parquet under `evaluation_outputs/` — read-only for parity.
- Full `ê` decomposition with aleatoric `a(x)` — available via `DEUPRanker(decompose=True)`
  but not part of original Chapter 13 g training.

See **P12** for the reproducible benchmark suite including finance walk-forward and
CIFAR-10-C aggregation.
