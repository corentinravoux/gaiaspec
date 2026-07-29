# gaiaspec

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21646138.svg)](https://doi.org/10.5281/zenodo.21646138)

Turn [Gaia](https://www.cosmos.esa.int/web/gaia) XP spectra into **standard-star
spectra** for atmospheric calibration of AuxTel (and other instruments), in the
same [CALSPEC](https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec)
dictionary format as the reference standards.

📖 **Full documentation (API reference + user guide):** **<https://corentinravoux.github.io/gaiaspec/>**

---

## What it does

```
star name / Gaia id ─▶ getGaia            resolve id + load XP coefficients
                    ─▶ gaiaxpy calibrate  flux vs wavelength
                    ─▶ spectrum_correction empirical flux correction
                    ─▶ CALSPEC-format spectrum (WAVELENGTH/FLUX/STATERROR/SYSERROR)

Gaia archive ─▶ standard  build the bundled source / spectra parquet tables
```

## Install

```bash
pip install .              # deps: pandas, numpy, scipy, matplotlib, astropy,
                           #       gaiaxpy, pyarrow, fastparquet
pip install -e ".[dev]"    # + pytest, to run the test suite
```

The archive/resolution features additionally use
[`getCalspec`](https://github.com/LSSTDESC/getCalspec) and `astroquery`. The
bundled tables under `gaiaspec/data/` let most of the package work offline.

## Quick start

```python
from gaiaspec import getGaia

star = getGaia.Gaia("HD111980")                 # resolve via Simbad -> Gaia DR3
spec = star.get_spectrum_numpy(flux_correction="m3")
wavelength = spec["WAVELENGTH"]                 # astropy Quantity (nm)
flux       = spec["FLUX"]                        # W m^-2 nm^-1
```

Apply a flux correction to your own Gaia flux:

```python
import numpy as np
from gaiaspec import spectrum_correction

correction = spectrum_correction.return_gaia_spectra_correction(
    wavelength, flux, corr_threshold=0.8, choose_corr="m3",
)
corrected_flux = flux * correction
```

## Package map

| Module | Role |
|---|---|
| `getGaia` | Resolve a star, load/download its Gaia XP spectrum, calibrate it and format it as a CALSPEC dict; the `Gaia` standard-star class |
| `spectrum_correction` | Correlate a spectrum against bundled Gaia/CALSPEC pairs and return the empirical flux correction |
| `standard` | Query the Gaia archive around sky fields, select by quality flag / spectral type, download spectra and write the reference tables |
| `utils` | Small internal helpers |

`gaiaspec/data/` holds the bundled Gaia source / spectra tables, the CALSPEC
matching table and the flux-correction table. `scripts/` holds the DDF-catalog
builder.

## Conventions

- Wavelengths in **nm**, fluxes in **W m⁻² nm⁻¹** (astropy `Quantity`).
- A spectrum is a dict with keys `WAVELENGTH`, `FLUX`, `STATERROR`, `SYSERROR`
  (CALSPEC convention), so gaiaspec spectra are drop-in standards.

## Documentation

📖 **Online: <https://corentinravoux.github.io/gaiaspec/>**

Built from the in-code docstrings with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) +
[mkdocstrings](https://mkdocstrings.github.io/) and deployed to GitHub Pages on
every push to `main`. Build locally with:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

See [`test/README.md`](test/README.md) for the layout of the suite.
