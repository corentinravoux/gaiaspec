# Getting started

## Installation

```bash
git clone https://github.com/corentinravoux/gaiaspec.git
cd gaiaspec
pip install .            # or: pip install -e ".[dev]" for tests
```

Core dependencies (`pandas`, `numpy`, `scipy`, `matplotlib`, `astropy`,
`gaiaxpy`, `pyarrow`, `fastparquet`) install automatically. Two extra packages
are needed for the archive/resolution features of [`getGaia`](api/getGaia.md):

- [`getCalspec`](https://github.com/LSSTDESC/getCalspec) — CALSPEC lookup;
- `astroquery` — Simbad / Gaia queries.

The bundled parquet / CSV tables under `gaiaspec/data/` let most of the package
work fully offline.

## Fetching a standard spectrum

```python
from gaiaspec import getGaia

# by star name (resolved to a Gaia DR3 id via Simbad)
star = getGaia.Gaia("HD111980")
spec = star.get_spectrum_numpy()          # CALSPEC dictionary format
print(spec.keys())                        # WAVELENGTH, FLUX, STATERROR, SYSERROR

# apply the empirical flux correction
spec_corr = star.get_spectrum_numpy(flux_correction="m3", correction_threshold=0.8)
```

Check availability before building a `Gaia` object:

```python
getGaia.is_gaiaspec("HD111980")   # is the star in the bundled catalog?
getGaia.is_gaia_full("HD111980")  # can a spectrum be retrieved at all?
```

## Working offline with the bundled tables

```python
from gaiaspec import getGaia, spectrum_correction

sources  = getGaia.get_gaia_sources()            # bundled source catalog
spectra  = getGaia.get_gaia_spectra()            # bundled XP coefficients
matching = getGaia.get_gaia_calspec_matching()   # CALSPEC <-> Gaia names

corr_table = spectrum_correction.load_gaia_corrections()
```

## Applying a flux correction directly

```python
import numpy as np
from gaiaspec import spectrum_correction

wavelength = np.linspace(400, 900, 200)   # nm
flux = my_gaia_flux                       # W m^-2 nm^-1
correction = spectrum_correction.return_gaia_spectra_correction(
    wavelength, flux, corr_threshold=0.8, choose_corr="m3",
)
corrected_flux = flux * correction
```

## Building a new catalog

Building catalogs queries the Gaia archive (requires login) — see
[`standard`](api/standard.md):

```python
from gaiaspec import standard

standard.login()
fields = standard.get_ddf()   # LSST Deep Drilling Field centres
source_table, spectra_table, types = standard.create_standard_catalog_around_fields(
    fields["RA"], fields["DEC"], delta_ra=3.0, delta_dec=3.0,
    mag_max=8.0, name="gaia_ddf", star_type_selection=["A", "G"],
)
```
