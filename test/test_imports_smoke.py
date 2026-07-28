"""Smoke tests: every shipped module byte-compiles and imports."""

import importlib
import py_compile

import pytest

from conftest import PACKAGE_ROOT

MODULES = ["utils", "spectrum_correction", "getGaia", "standard"]

ALL_PY = sorted((PACKAGE_ROOT / "gaiaspec").glob("*.py")) \
    + sorted((PACKAGE_ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("path", ALL_PY, ids=[p.name for p in ALL_PY])
def test_module_compiles(path):
    """Every shipped .py file byte-compiles (no syntax errors)."""
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    """Each package module imports (the suite is skipped if deps are missing)."""
    importlib.import_module(f"gaiaspec.{module}")
