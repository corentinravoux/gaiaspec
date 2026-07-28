"""Unit tests for :mod:`gaiaspec.utils`."""

import os

from gaiaspec import utils


def test_get_package_dir_points_at_package():
    d = utils._getPackageDir()
    assert os.path.isdir(d)
    # the bundled data directory lives inside the package
    assert os.path.isdir(os.path.join(d, "data"))
