"""Unit tests for the offline parts of :mod:`gaiaspec.getGaia`."""

import numpy as np
from astropy import units as u

from gaiaspec import getGaia


# --- bundled-data loaders --------------------------------------------------


def test_get_gaia_sources_and_spectra():
    sources = getGaia.get_gaia_sources()
    spectra = getGaia.get_gaia_spectra()
    assert len(sources) > 0
    assert "source_id" in sources.columns
    assert "source_id" in spectra.columns


def test_get_gaia_calspec_matching():
    df = getGaia.get_gaia_calspec_matching()
    for col in ("Star_name", "GAIA_DR3_Name"):
        assert col in df.columns


# --- pure conversion -------------------------------------------------------


def test_convert_to_calspec_format_units():
    wavelength = np.linspace(400, 900, 10)
    flux = np.ones(10)
    err = np.full(10, 0.1)
    spec = getGaia.convert_gaia_spectrum_to_calspec_format(wavelength, flux, err)
    assert set(spec.keys()) == {"WAVELENGTH", "FLUX", "STATERROR", "SYSERROR"}
    assert spec["WAVELENGTH"].unit == u.nm
    assert spec["FLUX"].unit == u.W / u.m**2 / u.nm
    # no correction requested -> flux passes through unchanged
    np.testing.assert_allclose(spec["FLUX"].to_value(), flux)
    # systematic error is zero-initialised
    np.testing.assert_allclose(spec["SYSERROR"].to_value(), 0.0)


def test_convert_to_calspec_format_with_correction():
    wavelength = np.linspace(400, 900, 20)
    flux = np.ones(20)
    err = np.full(20, 0.1)
    spec = getGaia.convert_gaia_spectrum_to_calspec_format(
        wavelength, flux, err, flux_correction="m3", correction_threshold=0.8
    )
    # a correction was applied -> still finite, same shape
    assert spec["FLUX"].shape == flux.shape
    assert np.all(np.isfinite(spec["FLUX"].to_value()))


# --- cache-name helper -----------------------------------------------------


def test_get_cache_file_sanitises_label():
    assert getGaia._get_cache_file("HD 111980") == "HD_111980"
    assert getGaia._get_cache_file("* alf Lyr") == "_alf_Lyr"
    assert getGaia._get_cache_file("Gaia.DR3") == "Gaia_DR3"
