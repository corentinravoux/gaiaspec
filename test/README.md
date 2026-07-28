# gaiaspec test suite

Run from the package root:

```bash
pip install -e ".[dev]"   # installs pytest alongside the runtime deps
pytest -q                 # or: pytest test/
```

The suite exercises the many **offline** functions against the bundled
`gaiaspec/data` tables — no network access is required. It skips itself
gracefully if the package cannot be imported (see the note below).

| File | Scope | What it covers |
|---|---|---|
| `test_utils.py` | unit | `_getPackageDir` resolves the package directory containing `data/` |
| `test_spectrum_correction.py` | unit + e2e | `multi_pearsonr` vs `scipy.stats.pearsonr`; loading the bundled correction table; a reference spectrum correlating perfectly with itself; `find_best_gaia_correction` / `return_gaia_spectra_correction(_multiple)` end to end |
| `test_getGaia.py` | unit | bundled source / spectra / matching loaders; `convert_gaia_spectrum_to_calspec_format` units and correction path; `_get_cache_file` label sanitisation |
| `test_standard.py` | unit | `get_ddf` fields; `chunks_data_split`; the pure-pandas `catalog_selection` Gaia-flag filters (RUWE, variability, processing mode) and its `ValueError` guards |
| `test_imports_smoke.py` | smoke | every shipped module byte-compiles and imports |

## Note on optional dependencies

`gaiaspec/__init__` eagerly imports `getGaia`, which imports `gaiaxpy` and
`getCalspec`. When those optional packages are absent the whole suite is
**skipped** (via `pytest.importorskip`) rather than failing — install them
(`pip install gaiaxpy getCalspec`) to run the offline tests. The network-only
functions (Gaia/Simbad queries, archive downloads) are intentionally not
exercised.
