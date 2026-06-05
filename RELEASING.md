# Releasing

## Version policy

We follow [SemVer](https://semver.org/). Tag releases as `vX.Y.Z` (e.g. `v0.3.1`).

Update `version` in `pyproject.toml` and `CHANGELOG.md` before tagging.

## PyPI (trusted publishing)

The release workflow (`.github/workflows/release.yml`) publishes to PyPI via
**OIDC trusted publishing**. One-time setup at
[pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/):

| Field | Value |
|---|---|
| PyPI project name | `deup` |
| Owner | `ursinasanderink` |
| Repository | `deup` |
| Workflow name | `Release` |
| Environment name | `pypi` |

Create a matching GitHub Environment named **`pypi`** (Settings → Environments) if you
use environment protection rules.

### Dry-run (no upload)

Actions → **Release** → **Run workflow** with *Build wheel only* checked. This runs
the build + smoke-install job without publishing.

## Cut a release

```bash
# 1. Bump version in pyproject.toml + CHANGELOG.md
# 2. Commit and push to main
git tag v0.3.1
git push origin v0.3.1
```

The workflow will:

1. Build sdist + wheel
2. Smoke-install the wheel in a fresh venv
3. Publish to PyPI (tag pushes only)
4. Create a GitHub Release with attached artifacts

## Verify after publish

```bash
python -m venv /tmp/verify-deup
source /tmp/verify-deup/bin/activate
pip install deup==0.3.1
python -c "from deup import DEUPRegressor; print('ok')"
```

## GitHub Pages (docs)

Docs deploy automatically on push to `main` via `.github/workflows/docs.yml`.

Enable once: **Settings → Pages → Build and deployment → GitHub Actions**.

Site: https://ursinasanderink.github.io/deup/

## TestPyPI (optional)

For pre-release testing, add a second trusted publisher on
[test.pypi.org](https://test.pypi.org/manage/account/publishing/) and a manual job
that uploads with `repository-url: https://test.pypi.org/legacy/`. Not required for
normal releases once PyPI trusted publishing is configured.

## Manual upload (fallback)

```bash
python -m pip install build twine
python -m build
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-XXXX twine upload dist/deup-*
```

## Zenodo DOI (optional)

Enable the [Zenodo GitHub integration](https://docs.zenodo.org/guides/github/) on the
`ursinasanderink/deup` repository to mint a DOI per release. Update `CITATION.cff` with
the DOI once reserved.

## Troubleshooting failed releases

| Symptom | Likely cause | Fix |
|---|---|---|
| PyPI 403 on first tag | Trusted publishing not configured | Complete PyPI publisher setup; retry with a new patch tag |
| Pages deploy failed | Pages source not set to GitHub Actions | Settings → Pages → GitHub Actions |
| `environment: pypi` pending | Environment requires approval | Approve in Actions or relax environment rules |

**Historical note:** `v0.1.0` failed to publish because trusted publishing was not yet
configured. `v0.1.1` and `v0.3.0` succeeded afterward. Stale "failed deployment"
badges for `v0.1.0` in the GitHub UI can be ignored.
