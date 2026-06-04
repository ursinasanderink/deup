# Architecture

This document captures the load-bearing design decisions for `deup`. It is the
contract that keeps the library general without becoming a god-object, and honest
about time-series correctness.

## 1. DEUP is a meta-algorithm, not a model

DEUP wraps *any* predictor: train `f`, collect `f`'s out-of-sample errors, train a
secondary predictor `g` to estimate those errors, expose `g(x)` as epistemic
uncertainty (optionally minus an aleatoric estimate `a(x)`). We therefore do **not**
extend PyTorch or any framework — we orchestrate models behind a small,
scikit-learn-style protocol (`fit` / `predict` / `predict_proba`). PyTorch is an
**optional backend** (`deup[torch]`), never the foundation.

## 2. The five axes (every use case is a configuration)

All supported use cases differ only along five pluggable axes; the core orchestration
is identical:

| Axis | Strategy object | Examples |
|---|---|---|
| 1. Task | estimator class | regression, classification, ranking, quantile |
| 2. Loss / error target | `Loss` | squared, log-loss, pinball, rank-loss, callable |
| 3. Grouping | `group_by` | i.i.d. rows, panel-by-entity, cross-section-by-date |
| 4. Out-of-sample scheme | `cv` splitter | KFold, GroupKFold, TimeSeriesSplit, PurgedWalkForward |
| 5. `g`-features | feature pipeline | raw X, density, variance, distance-to-train |

Use-case map:

| Use case | task | loss | group | cv | g-features |
|---|---|---|---|---|---|
| Cross-sectional ranker | ranking | rank-loss | by-date | PurgedWalkForward | score, vol, regime |
| Mean-reversion forecast | regression | squared | time | TimeSeriesSplit | residual, vol |
| Direction / credit | classification | log-loss | time / iid | walk-forward / Stratified | density, margin |
| Quantile / vol | quantile | pinball | time | walk-forward | realized-vol |
| OOD / vision | classification | per-sample loss | iid | holdout + seen-bit | embedding density, GP var |
| Active learning / BO | any | predicted error | iid | KFold | density, distance |
| Generic tabular | reg / clf | squared / log-loss | iid | KFold | raw X, density |

## 3. Layered primitives + thin wrappers

Build the primitives, then ship convenience estimators over them:

- `OOFErrorCollector(estimator, cv, loss, group_by)` — leakage-correct out-of-fold
  errors (the crux).
- feature builders + pipeline — what `g` sees.
- `ErrorEstimator(model, features)` — fits `g`.
- `UncertaintyCalibrator` — turns relative `g(x)` into calibrated intervals (v0.2+).
- `DEUPRegressor` / `DEUPClassifier` / `DEUPRanker` — ~20–40 line wrappers composing
  the above, with the ergonomic `predict(X, return_uncertainty=True)` API.

## 4. General core, time-series flagship

The core is splitter-agnostic and i.i.d.-clean, so the general crowd gets a simple,
correct API. But leakage-control is **first-class**: `PurgedWalkForward` /
`EmbargoedKFold` ship in the core with dedicated leakage tests, because correct
out-of-fold error construction for sequential / cross-sectional data is the
differentiator versus vision-centric UQ frameworks. Marketing leads with time-series;
the abstractions stay general.

## 5. Non-negotiable: no leakage

Every fold-local quantity (the error targets, scalers, density references, aleatoric
estimates) is fit on training folds only, inside the CV loop. A future-peeking
splitter must make a designed test fail. This is enforced in code, not assumed.

## 6. Attribution

DEUP the *method* is Lahlou, Jain, Nekoei, Butoi, Bertin, Rector-Brooks, Korablyov,
and Bengio (2023, TMLR). This repository is an independent library implementation;
it credits the method and does not claim it.
