# Tutorial: Cross-sectional finance ranking (flagship)

**Goal:** uncertainty for a panel ranker (one row per date × asset) with walk-forward
CV, rank-geometry decoupling, and context-level health gating.

## Panel format

Long-format DataFrame: one row per `(date, asset)` with features, score, and target.

```python
import pandas as pd
from deup.domains.finance import CrossSectionalDEUP

# panel_df columns: date, score, vol_20d, market_vol_21d, target_20d, ...
model = CrossSectionalDEUP(
    horizon=20,       # uses target_20d when present
    cv=5,
    embargo=1,
    random_state=0,
)
model.fit(panel_df)
```

Defaults wired in: `DEUPRanker` + `PurgedWalkForward` + rank residualization +
`HealthIndex` + finance g-feature preset.

## Predict

```python
pred, unc = model.predict(test_panel, return_uncertainty=True)
```

Pass the same `date` column via the panel — residualization is applied per date.

## Calibrated intervals (held-out calibration split)

```python
model.calibrate(cal_panel, alpha=0.1)
interval = model.predict_interval(test_panel)
interval.lower, interval.upper
```

Use a calibration panel the model did **not** see during `fit`.

## Context-level health (Finding 2; Sanderink, 2026)

Do **not** rely on `mean(g)` per day at N≈50 — see
[Aggregation reliability](../reliability.md) (Sanderink, 2026, Finding 1).

```python
report = model.health_report(test_panel)
report.health    # per-date score in [0, 1]
report.gate      # True = trust / trade this date
```

## Walk-forward on pre-computed residuals

If you already have walk-forward scores and realized rank losses in a panel (Step 0
done elsewhere), you can train only the error predictor `g`:

```python
from deup.domains.finance_walkforward import walkforward_g_on_enriched

preds, diagnostics = walkforward_g_on_enriched(enriched_df, horizons=[20])
```

See the API reference for `walkforward_g_on_enriched`.

## Next

- [Concepts: five axes](../concepts.md)
- [Benchmarks: finance walk-forward](../benchmarks.md)
