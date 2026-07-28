"""Top-level package for gaiaspec.

Exposes :mod:`gaiaspec.getGaia` (retrieve and calibrate Gaia XP spectra as
CALSPEC-style standards) and :mod:`gaiaspec.standard` (build standard-star
catalogs from the Gaia archive).
"""

import os

from . import getGaia, standard

__version__ = "1.0.2"
__gaiaspec_dir_path__ = os.path.dirname(__file__)
