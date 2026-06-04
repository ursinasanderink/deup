# Losses & target transforms

DEUP trains a secondary predictor `g` on the base model's **pointwise error**
`l(y, f(x))`. Choose the loss that matches your task.

## Registry

| Name | Task | Formula (per row) |
|---|---|---|
| `squared` | regression | `(y - ŷ)²` |
| `absolute` | regression | `\|y - ŷ\|` |
| `pinball` / `pinball:0.9` | quantile regression | pinball at quantile `q` |
| `logloss` | classification | cross-entropy |
| `brier` | classification | Brier score |
| `rank` | cross-sectional ranking | `\|rank_pct(y) - rank_pct(ŷ)\|` per group |

```python
from deup.core import get_loss

loss_fn = get_loss("brier")
loss_fn = get_loss("pinball", q=0.9)
loss_fn = get_loss("pinball:0.75")
```

Pass a custom callable to `get_loss` or directly to `DEUPRegressor(loss=...)`.

## Target transforms for `g`

Error targets can be heavy-tailed. Before fitting `g`, apply:

| Transform | Forward | Inverse |
|---|---|---|
| `log` (default) | `log(err + ε)` | `exp(g) - ε` |
| `asinh` | `asinh(err / ε)` | `sinh(g) × ε` |
| `none` | identity | identity |

```python
from deup.core import apply_error_transform, inverse_error_transform

t = apply_error_transform(errors, method="asinh", eps=1.0)
back = inverse_error_transform(t, method="asinh", eps=1.0)
```

In `DEUPRegressor`, set `target_transform="log" | "asinh" | "none"`.

## Task → config cheat sheet

| Use case | `loss` | `cv` | `groups` |
|---|---|---|---|
| Tabular regression | `squared` | `5` (KFold) | — |
| Quantile forecast | `pinball:0.9` | `TimeSeriesSplit` | — |
| Classification (future) | `logloss` / `brier` | `StratifiedKFold` | — |
| Cross-sectional ranker | `rank` | `PurgedWalkForward` | dates |
