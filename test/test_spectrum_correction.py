"""Unit / end-to-end tests for :mod:`gaiaspec.spectrum_correction`."""

import numpy as np
import pytest
from scipy.stats import pearsonr

from gaiaspec import spectrum_correction


def test_multi_pearsonr_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 60))
    y = rng.normal(size=60)
    got = spectrum_correction.multi_pearsonr(x, y)
    expected = np.array([pearsonr(x[i], y)[0] for i in range(5)])
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_multi_pearsonr_clipped():
    # a perfectly correlated row gives exactly 1
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    got = spectrum_correction.multi_pearsonr(x, y)
    assert got[0] == pytest.approx(1.0)


def test_load_gaia_corrections_columns(correction_table):
    for col in ("wavelength", "gaia_spectrum", "gaia_corrected_flux_m3"):
        assert col in correction_table.columns
    assert len(correction_table) > 0


def test_compute_gaia_correlation_self(correction_table, reference_spectrum):
    wavelength, flux = reference_spectrum
    corr = spectrum_correction.compute_gaia_correlation(
        correction_table, wavelength, flux
    )
    # the reference correlates perfectly with itself
    assert np.nanmax(corr) == pytest.approx(1.0, abs=1e-6)
    assert len(corr) == len(correction_table)


def test_return_gaia_spectra_correction_shape(reference_spectrum):
    wavelength, flux = reference_spectrum
    correction = spectrum_correction.return_gaia_spectra_correction(
        wavelength, flux, corr_threshold=0.8, choose_corr="m3",
    )
    assert correction.shape == wavelength.shape
    assert np.all(np.isfinite(correction))


def test_find_best_correction_below_threshold_returns_ones(correction_table):
    # a flat wavelength grid with an impossible threshold -> unit correction
    table = correction_table.copy()
    table["correlation_with_gaia"] = np.zeros(len(table))
    wavelength = np.linspace(400, 900, 50)
    corr = spectrum_correction.find_best_gaia_correction(
        table, wavelength, corr_threshold=0.9, verbose=False
    )
    np.testing.assert_array_equal(corr, np.ones_like(wavelength))


def test_return_gaia_spectra_correction_multiple(reference_spectrum):
    wavelength, flux = reference_spectrum
    corrections = spectrum_correction.return_gaia_spectra_correction_multiple(
        wavelength, [flux, flux], corr_threshold=0.8, choose_corr="m3",
    )
    assert corrections.shape[0] == 2
    assert corrections.shape[1] == wavelength.size
