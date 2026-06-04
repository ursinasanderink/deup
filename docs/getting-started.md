# Getting started

## Install

```bash
pip install deup
```

Optional extras:

```bash
pip install "deup[gbm]"    # LightGBM error predictor
pip install "deup[torch]"    # neural / GP backends (v0.2+)
pip install "deup[docs]"     # MkDocs site locally
```

## Quickstart — tabular regression

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deup import DEUPRegressor

X, y = fetch_california_housing(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

model = DEUPRegressor(
    base_model=RandomForestRegressor(n_estimators=100, random_state=0),
    cv=5,
    random_state=0,
)
model.fit(X_tr, y_tr)

pred, unc = model.predict(X_te, return_uncertainty=True)
```

- `pred` — point prediction from the base model (refit on all training data)
- `unc` — estimated epistemic uncertainty `g(x)` (predicted out-of-sample error)

Higher `unc` means the model is likely to be wrong at that input. On California
housing (v0.1 benchmark), Spearman(`unc`, realized squared error) ≈ **0.51**, beating
ensemble disagreement (0.46) and a conformal residual baseline (0.45).

## Time-series / walk-forward

For ordered or panel data, pass a leakage-safe splitter:

```python
from deup import DEUPRegressor
from deup.splitters import PurgedWalkForward

model = DEUPRegressor(
    base_model=my_model,
    cv=PurgedWalkForward(n_splits=5, embargo=5, min_train_size=20),
)
model.fit(X, y, groups=dates)  # dates = cross-section label per row
pred, unc = model.predict(X_test, return_uncertainty=True)
```

`PurgedWalkForward` keeps each date's full cross-section together and drops an
**embargo** between train and test to prevent look-ahead leakage.

## Cross-sectional ranking

Use the `rank` loss with group-coherent CV:

```python
model = DEUPRegressor(
    base_model=my_ranker,
    loss="rank",
    cv=PurgedWalkForward(n_splits=5, embargo=5),
)
model.fit(X, y, groups=dates)
unc = model.predict_epistemic(X)
```

## Target stabilization

Heavy-tailed error targets are stabilized before training `g`:

```python
# default: log(error + eps)
model = DEUPRegressor(target_transform="log")

# robust alternative for very heavy tails
model = DEUPRegressor(target_transform="asinh")

# raw errors (no transform)
model = DEUPRegressor(target_transform="none")
```

## What v0.1 includes / excludes

**Included:** `DEUPRegressor`, leakage-correct OOF collection, splitters
(`KFold`, `TimeSeriesSplit`, `PurgedWalkForward`), loss registry, benchmark.

**Coming in v0.2:** `DEUPClassifier` / `DEUPRanker`, conformal intervals,
aleatoric decomposition, density/GP features, aggregation-reliability diagnostics.

## Run the benchmark locally

```bash
git clone https://github.com/ursinasanderink/deup.git
cd deup
pip install -e ".[dev]"
python benchmarks/run_regression_benchmark.py
```

Results land in `benchmarks/results/regression_benchmark.json`.
