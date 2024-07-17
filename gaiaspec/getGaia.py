import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
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


def is_gaia(label):
    try:
        label = int(label)
    except:
        return False
    gaia_sources = get_gaia_sources()
    return label in np.array(gaia_sources["SOURCE_ID"])


class Gaia:

    def __init__(
        self,
        label,
    ):
        try:
            label = int(label)
        except:
            raise ValueError(
                "The format of the given label is not appropriate for Gaia."
            )
        self.label = label

        gaia_sources = get_gaia_sources()
        mask = np.array(gaia_sources["SOURCE_ID"]) == self.label
        if len(mask[mask]) < 1:
            raise KeyError(f"{self.label} not found in Gaia tables.")
        for col in gaia_sources.columns:
            setattr(self, col, gaia_sources[mask][col].values)
        self.wavelength = None
        self.flux = None
        self.stat = None
        self.syst = None

    def get_spectrum_numpy(self, type="stis", date="latest"):
        gaia_spectra = get_gaia_spectra()

        mask = np.array(gaia_spectra["source_id"]) == self.label
        calibrated_spectra, sampling = calibrate(gaia_spectra[mask])

        wavelength = sampling * u.nm
        gaia_flux = calibrated_spectra["flux"][0] * u.W / u.m**2 / u.nm
        gaia_flux_error = calibrated_spectra["flux_error"][0] * u.W / u.m**2 / u.nm
        gaia_flux_syserror = np.zeros_like(gaia_flux_error) * u.W / u.m**2 / u.nm
        
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
