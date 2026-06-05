# Launch post (draft)

Use/adapt for GitHub Discussions, LinkedIn, or a short blog post.

---

**Introducing `deup` — Direct Epistemic Uncertainty Prediction for scikit-learn**

Most uncertainty libraries give you ensembles, Bayesian posteriors, or conformal
intervals. **DEUP** (Lahlou et al., TMLR 2023) takes a different path: train a
secondary model to predict your base model's **out-of-sample error**. No retraining
the base model as a Bayesian net; works with Random Forest, LightGBM, or anything
with `fit`/`predict`.

Until now, the only public DEUP code was a research notebook repo with no `pip install`.
`deup` fills that gap:

- **Leakage-correct by default** — honest OOF error targets (Algorithm 2)
- **Time-series & cross-sectional finance** — purged walk-forward, rank residualization
- **Calibrated intervals** — split-conformal + MAPIE interop
- **Benchmarked** — on California housing, DEUP ranks realized error better than
  ensemble disagreement (Spearman **0.51** vs **0.46**)

```bash
pip install deup
```

```python
from sklearn.ensemble import RandomForestRegressor
from deup import DEUPRegressor

model = DEUPRegressor(base_model=RandomForestRegressor())
model.fit(X_train, y_train)
pred, unc = model.predict(X_test, return_uncertainty=True)
```

Docs: https://ursinasanderink.github.io/deup/  
Repo: https://github.com/ursinasanderink/deup

**Credit:** DEUP the method is due to Lahlou, Jain, Nekoei, Butoi, Bertin, Rector-Brooks,
Korablyov, and Bengio (2023). This package is an independent open-source implementation
and extension for tabular and time-series workflows.

If you use it in research, please cite both the software (`CITATION.cff`) and the original paper.
