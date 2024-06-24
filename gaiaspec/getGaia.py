import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits


def get_gaia_sources():
    dirname = _getPackageDir()
    filename = os.path.join(dirname, "../data/gaia_source_file.csv")
    df = pd.read_csv(filename)
    return df


def get_gaia_spectra():
    dirname = _getPackageDir()
    filename = os.path.join(dirname, "../data/gaia_spectrum_file.csv")
    df = pd.read_csv(filename)
    return df


def _getPackageDir():
    """This method must live in the top level of this package, so if this
    moves to a utils file then the returned path will need to account for that.
    """
    dirname = os.path.dirname(__file__)
    return dirname


def is_gaia(star_label):
    gaia_sources = get_gaia_sources()

    return np.any(get_calspec_keys(sanitizeString(star_label)))


class Calspec:

    def __init__(self, gaia_label):
        """

        Parameters
        ----------
        calspec_label: str
            The Simbad name of the calspec star

        Examples
        --------
        >>> c = Calspec("* eta01 Dor")
        >>> print(c)   #doctest: +ELLIPSIS
        eta1dor
        >>> c = Calspec("etta dor")   #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        KeyError: 'etta dor not found in Calspec tables.'
        >>> c = Calspec("mu col")
        >>> c = Calspec("* mu. Col")
        >>> print(c)
        mucol
        >>> c = Calspec("HD38666")
        >>> print(c)
        mucol
        """
        self.label = gaia_label
        test = is_calspec(self.label)
        if not test:
            raise KeyError(f"{calspec_label} not found in Calspec tables.")
        df = getCalspecDataFrame()
        row = df[get_calspec_keys(self.label)]
        self.query = row
        for col in row.columns:  # sets .STIS and .Name attributes
            setattr(self, col, row[col].values[0])
        self.wavelength = None
        self.flux = None
        self.stat = None
        self.syst = None

    def __str__(self):
        return self.Name

    def _sanitizeName(self, name):
        """Special casing for cleaning up names in the table for use in
        downloading.
        """
        name = name.lower()
        if name == "sdss151421":
            name = "sdssj151421"
        return name

    def get_file_dataframe(self, type="stis"):
        """Get the corresponding row from the history.csv table..

        Parameters
        ----------
        type: str
            Choose between STIS or model spectrum. Must be either 'stis'
            or 'mod' (default: 'stis').

        Returns
        -------
        row: pandas.DataFrame
            The row from the history.csv file.

        Examples
        --------
        >>> c = Calspec("2M0559-14")
        >>> row = c.get_file_dataframe(type="stis")
        >>> row


        """
        if type.lower() not in ["stis", "mod"]:
            raise ValueError(
                f"Type argument must be either 'stis' or 'mod'. Got {type=}."
            )
        versions = getHistoryDataFrame()
        versions.sort_values("Filename")  # ensure table is ordered in time
        rows = versions.loc[
            (versions["Name"] == self.Name)
            & (versions["Extension"].str.contains(type.lower()))
        ]
        rows.loc[:, "Date"] = pd.to_datetime(rows["Date"], format="mixed")
        return rows

    def get_spectrum_fits_filename(self, type="stis", date="latest"):
        """Get the file name extension of type 'mod' or 'stis' at the
        closest date before the given date.

        Parameters
        ----------
        type: str
            Choose between STIS or model spectrum. Must be either 'stis'
            or 'mod' (default: 'stis').
        date: str
            The most recent file before the given date will be returned
            (default: 'latest'). One can use all datetime formats understood
            by pandas `to_datetime()` method.

        Returns
        -------
        spectrum_file_name: str
            Spectrum file name in astropy cache folder.

        Examples
        --------
        >>> c = Calspec("10 lac")
        >>> c.get_spectrum_fits_filename(type="stis", date="latest")
        '10lac_stis_007.fits'
        >>> c.get_spectrum_fits_filename(type="mod", date="2021-03-20")
        '10lac_mod_003.fits'
        """
        if date == "latest":
            if type == "mod":
                extension = self.Model
            elif type == "stis":
                extension = self.STIS
        else:
            rows = self.get_file_dataframe(type=type)
            dt = pd.to_datetime(date)
            if dt < min(rows["Date"]):
                raise ValueError(
                    f"Given {date=} is lower than the oldest available date {min(rows['Date'])=}."
                )
            latest_row_before_date = rows.loc[max(rows[rows["Date"] <= dt].index)]
            extension = latest_row_before_date["Extension"]
        spectrum_file_name = (
            self._sanitizeName(self.Name) + extension.replace("*", "") + ".fits"
        )
        return spectrum_file_name

    def get_spectrum_table(self, type="stis", date="latest"):
        """

        Returns
        -------
        table: astropy.io.fits.FITS_rec
            FITS table containing all data for given Calspec star.

        Examples
        --------
        >>> c = Calspec("eta1 dor")
        >>> t = c.get_spectrum_table()
        >>> print([col.name for col in t.columns])   #doctest: +ELLIPSIS
        ['WAVELENGTH', 'FLUX', ...]
        >>> print([col.unit for col in t.columns])   #doctest: +ELLIPSIS
        ['ANGSTROMS', 'FLAM', ...]

        """
        output_file_name = self.download_spectrum_fits_filename(type=type, date=date)
        with warnings.catch_warnings():  # calspec fits files use non-astropy units everywhere
            warnings.filterwarnings("ignore", message=".*did not parse as fits unit")
            t = fits.getdata(output_file_name)
        return t

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
        tab = self.get_spectrum_table(type=type, date=date)
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
