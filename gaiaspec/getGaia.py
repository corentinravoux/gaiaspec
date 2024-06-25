import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from gaiaxpy import calibrate


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


def is_gaia(label):
    gaia_sources = get_gaia_sources()
    return label in np.array(gaia_sources["SOURCE_ID"])


class Gaia:

    def __init__(
        self,
        label,
    ):
        self.label = label

        gaia_sources = get_gaia_sources()
        mask = np.array(gaia_sources["SOURCE_ID"]) == self.label
        if len(mask[mask]) < 1:
            raise KeyError(f"{self.label} not found in Calspec tables.")
        for col in gaia_sources.columns:
            setattr(self, col, gaia_sources[mask][col].values)
        self.wavelength = None
        self.flux = None
        self.stat = None
        self.syst = None

    def get_spectrum_numpy(self, type="stis", date="latest"):
        """Make a dictionary of numpy arrays with astropy units from Calspec
        FITS file.

        Returns
        -------
        table: dict
            A dictionary with the FITS table columns and their astropy units.

        Examples
        --------
        >>> c = Calspec("1812524")
        >>> dict = c.get_spectrum_numpy()
        >>> print(dict)   #doctest: +ELLIPSIS
        {'WAVELENGTH': <Quantity [...

        """
        gaia_spectra = get_gaia_spectra()

        mask = np.array(gaia_spectra["SOURCE_ID"]) == self.label
        astropy_table = gaia_spectra[mask]
        d = {}
        ncols = len(tab.columns)
        for k in range(ncols):
            d[tab.columns[k].name] = np.copy(tab[tab.columns[k].name][:])
            if tab.columns[k].unit == "ANGSTROMS":
                d[tab.columns[k].name] *= u.angstrom
            elif tab.columns[k].unit == "NANOMETERS":
                d[tab.columns[k].name] *= u.nanometer
            elif tab.columns[k].unit == "FLAM":
                d[tab.columns[k].name] *= u.erg / u.second / u.cm**2 / u.angstrom
            elif tab.columns[k].unit == "SEC":
                d[tab.columns[k].name] *= u.second
        return d

    def plot_spectrum(self, xscale="log", yscale="log"):
        """Plot Calspec spectrum.

        Examples
        --------
        >>> c = Calspec("eta1 dor")
        >>> c.plot_spectrum()

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
