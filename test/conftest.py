"""Shared fixtures for the gaiaspec test suite.

The whole package is importable only when its heavy optional dependencies
(``gaiaxpy``, ``getCalspec``) are installed, because ``gaiaspec/__init__``
eagerly imports :mod:`gaiaspec.getGaia`. The suite therefore skips itself
gracefully when the package cannot be imported, and otherwise exercises the many
offline functions against the bundled ``gaiaspec/data`` tables — no network
access is required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

TEST_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
for _p in (str(PACKAGE_ROOT), str(TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Skip the entire suite if the package (and its heavy deps) cannot be imported.
gaiaspec = pytest.importorskip("gaiaspec")


@pytest.fixture(scope="session")
def correction_table():
    """The bundled Gaia flux-correction table."""
    from gaiaspec import spectrum_correction

    return spectrum_correction.load_gaia_corrections()


@pytest.fixture(scope="session")
def reference_spectrum(correction_table):
    """A ``(wavelength, flux)`` pair taken from the first correction-table row."""
    i = correction_table.index[0]
    wavelength = np.asarray(correction_table["wavelength"][i], dtype=float)
    flux = np.asarray(correction_table["gaia_spectrum"][i], dtype=float)
    return wavelength, flux


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run a test inside an isolated temporary working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
