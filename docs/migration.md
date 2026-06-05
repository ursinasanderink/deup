# Migrating from thesis code

See the full guide in the repository root: [MIGRATION.md](https://github.com/ursinasanderink/deup/blob/main/MIGRATION.md).

## Quick reference

**Before (thesis):**

```python
from src.uncertainty.deup_estimator import train_g_walk_forward

g_preds, diagnostics = train_g_walk_forward(enriched, target_col="rank_loss", min_train_folds=20)
```

**After (library, direct parity):**

```python
from deup.domains.finance_walkforward import walkforward_g_on_enriched

g_preds, diagnostics = walkforward_g_on_enriched(
    enriched, target_col="rank_loss", min_train_folds=20, fold_sort="numeric"
)
```

**After (library, end-to-end):**

```python
from deup.domains.finance import CrossSectionalDEUP

model = CrossSectionalDEUP(horizon=20).fit(panel_df)
```

## Parity result (Chapter 13 v3, H=20)

Exact match on 182,046 rows: max \|Δg\| = **0.0** vs frozen thesis `g_predictions_rank.parquet`.

Run locally:

```bash
pip install "deup[parity]"
python scripts/parity_thesis_finance.py --enriched ... --frozen-g ...
```
