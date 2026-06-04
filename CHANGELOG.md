# Changelog

## [0.1.0] — 2026-06-04

First public release.

### Added

- `DEUPRegressor` — sklearn-compatible wrapper with `predict(..., return_uncertainty=True)`
- Leakage-correct `OOFErrorCollector` (DEUP Algorithm 2 / K-fold OOF errors)
- Splitters: `PurgedWalkForward`, re-export `KFold` / `TimeSeriesSplit`
- Loss registry: `squared`, `absolute`, `logloss`, `brier`, `pinball`, `rank`
- Target transforms: `log`, `asinh`, `none` for error-predictor training
- Benchmark: DEUP vs ensemble vs conformal on California housing
- MkDocs documentation site
- 54+ unit tests including parity-exact OOF and leakage gate

### Notes

- Aleatoric decomposition (`ê = max(0, g - a)`), conformal intervals, and
  `DEUPClassifier` / `DEUPRanker` are planned for v0.2.

[0.1.0]: https://github.com/ursinasanderink/deup/releases/tag/v0.1.0
