"""Unit tests for the offline parts of :mod:`gaiaspec.standard`."""

import numpy as np
import pandas as pd
import pytest

from gaiaspec import standard


def test_get_ddf_fields():
    df = standard.get_ddf()
    assert list(df["field"]) == [
        "COSMOS", "ECDFS", "EDFS_a", "EDFS_b", "ELAISS1", "XMM_LSS"
    ]
    assert set(["field", "RA", "DEC"]).issubset(df.columns)
    assert len(df) == 6


def test_chunks_data_split():
    chunks = list(standard.chunks_data_split(list(range(7)), 3))
    assert chunks == [[0, 1, 2], [3, 4, 5], [6]]
    # exact multiple
    assert list(standard.chunks_data_split(list(range(6)), 2)) == [[0, 1], [2, 3], [4, 5]]


# --- catalog_selection (pure pandas filtering) -----------------------------


def _make_catalog():
    return pd.DataFrame(
        {
            "source_id": np.arange(6),
            "ruwe": [1.0, 1.5, 1.2, 2.0, 1.1, 1.3],
            "phot_variable_flag": ["CONSTANT", "VARIABLE", "NOT_AVAILABLE",
                                   "CONSTANT", "VARIABLE", "CONSTANT"],
            "phot_proc_mode": [0, 1, 2, 0, 1, 0],
            "phot_g_mean_mag": np.linspace(8, 12, 6),
        }
    )


def test_catalog_selection_ruwe():
    df = _make_catalog()
    out = standard.catalog_selection(df, sel_ruwe=True, drop_used_flag_columns=False)
    assert np.all(out["ruwe"] < 1.4)


def test_catalog_selection_variable_flag():
    df = _make_catalog()
    const = standard.catalog_selection(
        df, sel_variable_flag="CONSTANT", drop_used_flag_columns=False
    )
    assert not (const["phot_variable_flag"] == "VARIABLE").any()
    var = standard.catalog_selection(
        _make_catalog(), sel_variable_flag="VARIABLE", drop_used_flag_columns=False
    )
    assert (var["phot_variable_flag"] == "VARIABLE").all()


def test_catalog_selection_proc_mode_and_drop():
    df = _make_catalog()
    out = standard.catalog_selection(df, sel_proc_mode=0)
    # all kept rows had proc_mode 0, and the used flag column was dropped
    assert "phot_proc_mode" not in out.columns
    assert len(out) == 3


def test_catalog_selection_invalid_values():
    with pytest.raises(ValueError):
        standard.catalog_selection(_make_catalog(), sel_variable_flag="MAYBE")
    with pytest.raises(ValueError):
        standard.catalog_selection(_make_catalog(), sel_proc_mode=9)
    with pytest.raises(ValueError):
        standard.catalog_selection(_make_catalog(), sel_c_star=1)  # not a float
