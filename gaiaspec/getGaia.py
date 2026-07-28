"""Retrieve and calibrate Gaia XP spectra as CALSPEC-style standards.

Front-end of gaiaspec. Loads the bundled Gaia source / spectra tables, resolves
a star to its Gaia DR3 source id (via Simbad or the CALSPEC matching table),
calibrates the Gaia XP coefficients into flux-vs-wavelength spectra (via
``gaiaxpy``), optionally applies the empirical flux correction from
:mod:`gaiaspec.spectrum_correction`, and exposes it in the CALSPEC dictionary
format. The :class:`Gaia` class wraps a single standard star.
"""

import os
import shutil

import astropy.config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import ascii
from astroquery.simbad import SimbadClass
from gaiaxpy import calibrate
from getCalspec import getCalspec

from gaiaspec import spectrum_correction, utils

try:
    from spectractor.extractor.targets import _SIMBAD_VOTABLE_FIELDS
except ImportError:
    _SIMBAD_VOTABLE_FIELDS = (
        "U",
        "B",
        "V",
        "R",
        "I",
        "J",
        "sp_type",
        "parallax",
        "propermotions",
        "rvz_redshift",
        "IDS",
        "ids",
    )


def get_gaia_sources():
    """Load the bundled Gaia source catalog.

    Returns:
        pandas.DataFrame: The ``data/gaia_source_file.parquet`` source table.
    """
    dirname = utils._getPackageDir()
    filename = os.path.join(dirname, "./data/gaia_source_file.parquet")
    df = pd.read_parquet(filename)
    return df


def get_gaia_spectra():
    """Load the bundled Gaia XP spectra table (uncalibrated coefficients).

    Returns:
        pandas.DataFrame: The ``data/gaia_spectra_file.parquet`` table.
    """
    dirname = utils._getPackageDir()
    filename = os.path.join(dirname, "./data/gaia_spectra_file.parquet")
    df = pd.read_parquet(filename)
    return df


def get_gaia_spectra_calibrated(
    flux_correction=None,
    correction_threshold=0.8,
):
    """Calibrate every bundled Gaia XP spectrum into CALSPEC-format flux.

    Args:
        flux_correction (str, optional): Empirical correction variant to apply
            (e.g. ``"m3"``); None disables the correction.
        correction_threshold (float, optional): Minimum correlation for the
            correction to be applied.

    Returns:
        list[dict]: One CALSPEC-format spectrum dict per source.
    """
    df = get_gaia_spectra()
    calibrated_spectra, wavelength = calibrate(df)
    spectra = []
    for i in range(len(calibrated_spectra)):
        gaia_flux = calibrated_spectra["flux"][i]
        gaia_flux_error = calibrated_spectra["flux_error"][i]
        converted_spectra = convert_gaia_spectrum_to_calspec_format(
            wavelength,
            gaia_flux,
            gaia_flux_error,
            flux_correction=flux_correction,
            correction_threshold=correction_threshold,
        )
        spectra.append(converted_spectra)
    return spectra


def get_gaia_calspec_matching():
    """Load the bundled CALSPEC <-> Gaia name matching table.

    Returns:
        pandas.DataFrame: The ``data/calspec_gaia_matching.csv`` table with
        ``Star_name`` and ``GAIA_DR3_Name`` columns.
    """
    dirname = utils._getPackageDir()
    filename = os.path.join(dirname, "./data/calspec_gaia_matching.csv")
    df = pd.read_csv(filename)
    return df


def get_gaia_name_from_calspec(star_label):
    """Resolve a CALSPEC star label to its Gaia DR3 name.

    Args:
        star_label (str): CALSPEC star label / key.

    Returns:
        str: The matching ``GAIA_DR3_Name``.

    Raises:
        KeyError: If the star is not matched to exactly one Gaia entry.
    """
    df = getCalspec.getCalspecDataFrame()
    key = getCalspec.get_calspec_keys(star_label)
    calspec_star_name = df["Star_name"][key].iloc[0]
    df_matching = get_gaia_calspec_matching()
    mask = df_matching["Star_name"] == calspec_star_name
    if len(mask[mask]) != 1:
        raise KeyError(f"The star label {star_label} was not matched with gaia")
    return df_matching["GAIA_DR3_Name"][mask].iloc[0]


def get_gaia_from_query_id(
    source_id,
    output_path=".cache/gaiaxpy",
    truncation=False,
):
    """
    Retrieve Gaia spectrum data for one specified source ID.
    It can obtain the data from a CSV file located at the given 'path' or by calibrating the data with the specified 'wavelength_sampling'.

    Args:
        source_id (int): A source ID for which Gaia spectrum data is requested.
        wavelength_sampling (float): The desired wavelength sampling for the spectrums.
        path (str, optional): The path to a CSV file containing Gaia spectra data. Default is None.

    Returns:
        pd.DataFrame: A DataFrame containing the Gaia spectrums data for the specified source IDs.
    """
    os.makedirs(output_path, exist_ok=True)
    cache_catalog = os.path.join(output_path, "gaiaxpy_spectra.h5")
    sampling = os.path.join(output_path, "gaiaxpy_wls.npy")
    is_incatalog = False
    if os.path.isfile(cache_catalog):
        df = pd.read_hdf(cache_catalog)
        if source_id in list(df["source_id"]):
            wls = np.load(sampling)
            spec = df[df["source_id"] == source_id]
            is_incatalog = True
        else:
            spec, wls = download_spectrum_from_id(source_id, truncation=truncation)
            df = pd.concat([df, spec])
    else:
        spec, wls = download_spectrum_from_id(source_id, truncation=truncation)
        df = spec
    np.save(sampling, wls)
    if not is_incatalog:
        df.to_hdf(cache_catalog, key="df", mode="a")
    return wls, spec["flux"][0], spec["flux_error"][0]


def download_spectrum_from_id(
    source_id,
    truncation=False,
):
    """Download and calibrate a single Gaia XP spectrum from the Gaia archive.

    Args:
        source_id (int): Gaia DR3 source id.
        truncation (bool, optional): Apply gaiaxpy basis truncation.

    Returns:
        tuple: ``(df_spectrum, wavelength)`` — the calibrated spectrum dataframe
        and the wavelength sampling.
    """
    df_spectrum, wls = calibrate([source_id], truncation=truncation, save_file=False)
    return df_spectrum, wls


def convert_gaia_spectrum_to_calspec_format(
    wavelength,
    gaia_flux,
    gaia_flux_error,
    flux_correction=None,
    correction_threshold=0.8,
):
    """Wrap a calibrated Gaia spectrum in the CALSPEC dictionary format.

    Attaches astropy units, and optionally multiplies the flux and errors by the
    empirical correction from :mod:`gaiaspec.spectrum_correction`.

    Args:
        wavelength (numpy.ndarray): Wavelength grid (nm).
        gaia_flux (numpy.ndarray): Flux (W m^-2 nm^-1).
        gaia_flux_error (numpy.ndarray): Statistical flux error.
        flux_correction (str, optional): Correction variant (None disables it).
        correction_threshold (float, optional): Minimum correlation to apply it.

    Returns:
        dict: ``{"WAVELENGTH", "FLUX", "STATERROR", "SYSERROR"}`` with units.
    """
    wavelength = wavelength * u.nm
    gaia_flux = gaia_flux * u.W / u.m**2 / u.nm
    gaia_flux_error = gaia_flux_error * u.W / u.m**2 / u.nm
    gaia_flux_syserror = np.zeros(gaia_flux_error.shape) * u.W / u.m**2 / u.nm

    if flux_correction is not None:
        correction = spectrum_correction.return_gaia_spectra_correction(
            wavelength.to_value(),
            gaia_flux.to_value(),
            corr_threshold=correction_threshold,
            choose_corr=flux_correction,
        )
        gaia_flux = gaia_flux * correction
        gaia_flux_error = gaia_flux_error * correction
        gaia_flux_syserror = gaia_flux_syserror * correction

    return {
        "WAVELENGTH": wavelength,
        "FLUX": gaia_flux,
        "STATERROR": gaia_flux_error,
        "SYSERROR": gaia_flux_syserror,
    }


def _get_cache_dir():
    """Return (creating if needed) the Simbad query cache directory.

    Returns:
        str: The astroquery Simbad cache directory path.
    """
    cache = os.path.join(astropy.config.get_cache_dir(), "astroquery", "Simbad")
    os.makedirs(cache, exist_ok=True)
    return cache


def _get_cache_file(tag):
    """Sanitise a star label into a cache file basename.

    Args:
        tag (str): Star label.

    Returns:
        str: The label with ``*``/spaces/dots removed or replaced.
    """
    filename = tag.replace("*", "").replace(" ", "_").replace(".", "_")
    return filename


def _clean_cache_dir():
    """Delete the whole Simbad query cache directory."""
    cache = _get_cache_dir()
    shutil.rmtree(cache)


def get_gaia_name_from_star_name(
    label,
    debug=False,
):
    """
    Examples
    --------
    >>> id = get_gaia_name_from_star_name("HD111980")
    >>> id
    3510294882898890880
    """
    label_test = str(label)
    cache_location = _get_cache_dir()
    cache_file = f"{_get_cache_file(label_test)}.ecsv"
    if cache_file in os.listdir(cache_location):
        if debug:
            print(f"Using cached Simbad query for {label}")
        table = ascii.read(os.path.join(cache_location, cache_file))
    else:
        if debug:
            print(f"Querying Simbad for {label}")
        simbadQuerier = SimbadClass()
        simbadQuerier.add_votable_fields(*_SIMBAD_VOTABLE_FIELDS)
        table = simbadQuerier.query_object(label)
        table.write(os.path.join(cache_location, cache_file), overwrite=True)

    if table is None:
        if debug:
            print(f"No Simbad entry found for {label}")
        return None
    if debug:
        print(f"Table columns: {table.colnames}")
        print(f"Table contents: {table}")
    for col in ["IDS", "ids", "matched_id"]:
        if col in table.colnames:
            key_id = col
            break
    if debug:
        print(f"Using column '{key_id}' to extract Gaia ID")
    if len(list(table[key_id].data)) == 0:
        return None
    ids = list(table[key_id].data)[0].split("|")
    if debug:
        print(f"IDs found for {label}: {ids}")
    gaia_id = [ii for ii in ids if "Gaia DR3" in ii]
    if debug:
        print(f"Gaia IDs found for {label}: {gaia_id}")
    if gaia_id:
        gaia_id = int(gaia_id[0].split(" ")[-1])
    else:
        gaia_id = None
    return gaia_id


def is_gaiaspec(label, debug=False):
    """Whether a star is present in the bundled Gaia source catalog.

    Resolves the label to a Gaia DR3 id (via Simbad) if needed, then checks the
    bundled source table.

    Args:
        label (str | int): Star name or Gaia source id.
        debug (bool, optional): Print resolution diagnostics.

    Returns:
        bool: True if the star is in the bundled catalog.
    """
    test_gaia_name = get_gaia_name_from_star_name(label, debug=debug)
    if test_gaia_name is not None:
        label = test_gaia_name
    gaia_sources = get_gaia_sources()
    return label in np.array(gaia_sources["source_id"])


def is_gaia_full(label, debug=False):
    """Whether a Gaia XP spectrum can be retrieved for a star.

    Resolves the label to a Gaia DR3 id and tries to fetch its spectrum from the
    archive/cache.

    Args:
        label (str | int): Star name or Gaia source id.
        debug (bool, optional): Print resolution diagnostics.

    Returns:
        bool: True if a spectrum is available.
    """
    test_gaia_name = get_gaia_name_from_star_name(label, debug=debug)
    if test_gaia_name is not None:
        label = test_gaia_name
    else:
        return False
    try:
        gaia_spectrum = get_gaia_from_query_id(label)
        if debug:
            print(f"Gaia spectrum found for {label}")
        if gaia_spectrum is not None:
            return True
        else:
            return False
    except ValueError as ve:
        if debug:
            print(f"ValueError: {ve}")
            print(f"No Gaia spectrum found for {label}")
        return False


class Gaia:
    """A single Gaia standard star and its calibrated XP spectrum.

    Resolves ``label`` to a Gaia DR3 source id, copies the matching source-table
    columns onto the instance as attributes, and lazily exposes the calibrated
    (optionally flux-corrected) spectrum.
    """

    def __init__(
        self,
        label,
    ):
        """Resolve the star and attach its source-catalog metadata.

        Args:
            label (str | int): Star name or Gaia source id.
        """
        test_gaia_name = get_gaia_name_from_star_name(label)
        if test_gaia_name is not None:
            label = test_gaia_name
        self.label = label

        gaia_sources = get_gaia_sources()
        mask = np.array(gaia_sources["source_id"]) == self.label
        if len(mask[mask]) >= 1:
            for col in gaia_sources.columns:
                setattr(self, col, gaia_sources[mask][col].values)
        self.wavelength = None
        self.flux = None
        self.stat = None
        self.syst = None
        self.correction_flux = None

    def get_spectrum_numpy(
        self,
        flux_correction=None,
        correction_threshold=0.8,
    ):
        """Return the star's calibrated spectrum in CALSPEC dictionary format.

        Uses the bundled XP coefficients if the star is in the local table,
        otherwise downloads it from the Gaia archive.

        Args:
            flux_correction (str, optional): Correction variant (None disables).
            correction_threshold (float, optional): Minimum correlation to apply.

        Returns:
            dict: ``{"WAVELENGTH", "FLUX", "STATERROR", "SYSERROR"}`` with units.
        """
        gaia_spectra = get_gaia_spectra()

        mask = np.array(gaia_spectra["source_id"]) == self.label
        if len(mask[mask]) != 0:
            calibrated_spectra, wavelength = calibrate(gaia_spectra[mask])
            gaia_flux = calibrated_spectra["flux"][0]
            gaia_flux_error = calibrated_spectra["flux_error"][0]
        else:
            wavelength, gaia_flux, gaia_flux_error = get_gaia_from_query_id(self.label)

        return convert_gaia_spectrum_to_calspec_format(
            wavelength,
            gaia_flux,
            gaia_flux_error,
            flux_correction=flux_correction,
            correction_threshold=correction_threshold,
        )

    def plot_spectrum(self, xscale="log", yscale="log"):
        """Plot the star's calibrated flux vs wavelength with error bars.

        Args:
            xscale (str, optional): Matplotlib x-axis scale.
            yscale (str, optional): Matplotlib y-axis scale.
        """
        t = self.get_spectrum_numpy()
        _ = plt.figure()
        plt.errorbar(t["WAVELENGTH"].value, t["FLUX"].value, yerr=t["STATERROR"].value)
        plt.grid()
        plt.yscale(yscale)
        plt.xscale(xscale)
        plt.title(self.label)
        plt.xlabel(rf"$\lambda$ [{t['WAVELENGTH'].unit}]")
        plt.ylabel(rf"Flux [{t['FLUX'].unit}]")
        plt.show()
