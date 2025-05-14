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


def _getPackageDir():
    """This method must live in the top level of this package, so if this
    moves to a utils file then the returned path will need to account for that.
    """
    dirname = os.path.dirname(__file__)
    return dirname


def get_gaia_sources():
    dirname = _getPackageDir()
    filename = os.path.join(dirname, "./data/gaia_source_file.parquet")
    df = pd.read_parquet(filename)
    return df


def get_gaia_spectra():
    dirname = _getPackageDir()
    filename = os.path.join(dirname, "./data/gaia_spectra_file.parquet")
    df = pd.read_parquet(filename)
    return df


def get_gaia_calspec_matching():
    dirname = _getPackageDir()
    filename = os.path.join(dirname, "./data/calspec_gaia_matching.csv")
    df = pd.read_csv(filename)
    return df


def get_gaia_name_from_calspec(star_label):
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
    df_spectrum, wls = calibrate([source_id], truncation=truncation, save_file=False)
    return df_spectrum, wls


def _get_cache_dir():
    cache = os.path.join(astropy.config.get_cache_dir(), "astroquery", "Simbad")
    os.makedirs(cache, exist_ok=True)
    return cache


def _get_cache_file(tag):
    filename = tag.replace("*", "").replace(" ", "_").replace(".", "_")
    return filename


def _clean_cache_dir():
    cache = _get_cache_dir()
    shutil.rmtree(cache)


def get_gaia_name_from_star_name(label):
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
        table = ascii.read(os.path.join(cache_location, cache_file))
    else:
        simbadQuerier = SimbadClass()
        simbadQuerier.add_votable_fields("ids")
        table = simbadQuerier.query_object(label_test)
    if table is None:
        return None
    else:
        table.write(os.path.join(cache_location, cache_file), overwrite=True)
    try:
        ids = list(table["IDS"].data)[0].split("|")
    except:
        ids = list(table["ids"].data)[0].split("|")
    gaia_id = int([ii for ii in ids if "Gaia DR3" in ii][0].split(" ")[-1])
    return gaia_id


def is_gaiaspec(label):
    test_gaia_name = get_gaia_name_from_star_name(label)
    if test_gaia_name is not None:
        label = test_gaia_name
    gaia_sources = get_gaia_sources()
    return label in np.array(gaia_sources["source_id"])


def is_gaia_full(label):
    test_gaia_name = get_gaia_name_from_star_name(label)
    if test_gaia_name is not None:
        label = test_gaia_name
    try:
        gaia_spectrum = get_gaia_from_query_id(label)
        if gaia_spectrum is not None:
            return True
        else:
            return False
    except ValueError:
        return False


class Gaia:

    def __init__(
        self,
        label,
    ):
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

    def get_spectrum_numpy(self, type="stis", date="latest"):
        gaia_spectra = get_gaia_spectra()

        mask = np.array(gaia_spectra["source_id"]) == self.label
        if len(mask[mask]) != 0:
            calibrated_spectra, wavelength = calibrate(gaia_spectra[mask])
            gaia_flux = calibrated_spectra["flux"][0]
            gaia_flux_error = calibrated_spectra["flux_error"][0]
        else:
            wavelength, gaia_flux, gaia_flux_error = get_gaia_from_query_id(self.label)

        wavelength = wavelength * u.nm
        gaia_flux = gaia_flux * u.W / u.m**2 / u.nm
        gaia_flux_error = gaia_flux_error * u.W / u.m**2 / u.nm
        gaia_flux_syserror = np.zeros(gaia_flux_error.shape) * u.W / u.m**2 / u.nm

        return {
            "WAVELENGTH": wavelength,
            "FLUX": gaia_flux,
            "STATERROR": gaia_flux_error,
            "SYSERROR": gaia_flux_syserror,
        }

    def plot_spectrum(self, xscale="log", yscale="log"):
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
