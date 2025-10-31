import os

import numpy as np
import pandas as pd

from gaiaspec import utils


def load_gaia_corrections():
    dirname = utils._getPackageDir()
    filename = os.path.join(dirname, "./data/gaia_flux_correction.parquet")
    df_gaia_correction = pd.read_parquet(filename)
    return df_gaia_correction


def compute_gaia_correlation(
    df_gaia_correction,
    wavelength,
    flux,
):

    correlation_with_gaia = []

    for i in range(len(df_gaia_correction)):
        mask_nan_gaia_ref = ~np.isnan(df_gaia_correction["gaia_spectrum"][i])
        mask_nan_gaia = ~np.isnan(flux)

        flux_interp = np.interp(
            df_gaia_correction["wavelength"][i][mask_nan_gaia_ref],
            wavelength[mask_nan_gaia],
            flux[mask_nan_gaia],
        )

        correlation_with_gaia.append(
            np.corrcoef(
                flux_interp,
                df_gaia_correction["gaia_spectrum"][i][mask_nan_gaia_ref],
            )[0][1]
        )

    return np.array(correlation_with_gaia)


def find_best_gaia_correction(
    df_gaia_correction,
    wavelength,
    corr_threshold=0.8,
    choose_corr="m3",
    verbose=True,
):
    df_gaia_correction_sorted = df_gaia_correction.sort_values(
        "correlation_with_gaia", ascending=False
    )
    for i in df_gaia_correction_sorted.index:
        if df_gaia_correction_sorted["correlation_with_gaia"][i] < corr_threshold:
            if verbose:
                print("no GAIA correction was found under the selected criteria")
            return np.ones_like(wavelength)
        if (df_gaia_correction_sorted[f"correlation_with_gaia"][i] is not np.nan) and (
            df_gaia_correction_sorted[f"gaia_corrected_flux_{choose_corr}"][i]
            is not None
        ):
            correction_ref = (
                df_gaia_correction_sorted[f"gaia_corrected_flux_{choose_corr}"][i]
                / df_gaia_correction_sorted["gaia_spectrum"][i]
            )
            wave = df_gaia_correction_sorted["wavelength"][i]
            correction = np.interp(wavelength, wave, correction_ref)
            correction[np.isnan(correction)] = 1.0
            return correction
        else:
            return np.ones_like(wavelength)


def return_gaia_spectra_correction(
    wavelength,
    flux,
    corr_threshold=0.8,
    choose_corr="m3",
):
    df_gaia_correction = load_gaia_corrections()

    df_gaia_correction["correlation_with_gaia"] = compute_gaia_correlation(
        df_gaia_correction,
        wavelength,
        flux,
    )

    correction = find_best_gaia_correction(
        df_gaia_correction,
        wavelength,
        corr_threshold=corr_threshold,
        choose_corr=choose_corr,
    )
    return correction


def multi_pearsonr(x, y):
    xmean = x.mean(axis=1)
    ymean = y.mean()
    xm = x - xmean[:, None]
    ym = y - ymean
    normxm = np.linalg.norm(xm, axis=1)
    normym = np.linalg.norm(ym)
    return np.clip(np.dot(xm / normxm[:, None], ym / normym), -1.0, 1.0)


def compute_gaia_correlation_multiple_spectra(
    df_gaia_correction,
    wavelength,
    fluxes,
):

    df_gaia_correction_interpolated = np.zeros(
        (df_gaia_correction.shape[0], wavelength.size)
    )

    for i, index in enumerate(df_gaia_correction.index):
        wave_gaia_correction = df_gaia_correction["wavelength"][index]
        flux_gaia_correction = df_gaia_correction["gaia_spectrum"][index]

        mask_nan_gaia_correction = ~np.isnan(flux_gaia_correction)
        df_gaia_correction_interpolated[i] = np.interp(
            wavelength,
            wave_gaia_correction[mask_nan_gaia_correction],
            flux_gaia_correction[mask_nan_gaia_correction],
        )

    correlation_with_gaia = []

    for flux in fluxes:
        pearsonr_values = multi_pearsonr(df_gaia_correction_interpolated, flux)
        correlation_with_gaia.append(pearsonr_values)

    return np.array(correlation_with_gaia)


def return_gaia_spectra_correction_multiple(
    wavelength,
    fluxes,
    corr_threshold=0.8,
    choose_corr="m3",
):
    df_gaia_correction = load_gaia_corrections()

    correlation_with_gaia = compute_gaia_correlation_multiple_spectra(
        df_gaia_correction,
        wavelength,
        fluxes,
    )
    corrections = []
    for i, _ in enumerate(fluxes):
        df_gaia_correction["correlation_with_gaia"] = correlation_with_gaia[i]

        correction = find_best_gaia_correction(
            df_gaia_correction,
            wavelength,
            corr_threshold=corr_threshold,
            choose_corr=choose_corr,
            verbose=False,
        )
        corrections.append(correction)

    return np.array(corrections)
