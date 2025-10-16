import os

import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from astroquery.gaia import Gaia


def login():
    Gaia.login()


def get_spType(llist):
    """
    Function to get star spectral type from Simmbad

    Parameters
    ----------
    llist : list(str)
        List of star ids.

    Returns
    -------
    res : pandas df
        two cols: (target,spectral type).

    """

    from astroquery.simbad import Simbad

    simbad = Simbad()
    simbad.add_votable_fields("sptype")

    res = pd.DataFrame(llist, columns=["target"])
    tabb = simbad.query_objects(llist).to_pandas()
    res["spType"] = tabb["sp_type"]
    res = res.replace(r"^\s*$", "Unknown", regex=True)

    return res


def get_type_gaia(
    sourceidlist,
    Gaia_object,
    user="cravoux",
):
    table_id = Table([list(sourceidlist)], names=["gaia_id"], meta={"meta": "table"})
    Gaia_object.upload_table(upload_resource=table_id, table_name="tableid")

    query = f"SELECT ap.* \
    FROM gaiadr3.astrophysical_parameters AS ap \
    JOIN user_{user}.tableid as usert ON usert.gaia_id = ap.source_id"

    # launch query and save and return a Dataframe
    job = Gaia_object.launch_job_async(query)
    results = job.get_results().to_pandas()

    # suggest delete your table from your personal space
    Gaia_object.delete_user_table(table_name="tableid")

    df_types = results[
        [
            "teff_gspphot",
            "teff_gspspec",
            "teff_gspspec_lower",
            "teff_gspspec_upper",
            "teff_espucd",
            "teff_msc1",
            "teff_msc2",
            "spectraltype_esphs",
        ]
    ]
    return df_types


def get_ddf():
    """
    To get DDFs (name and position)

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    fields = ["COSMOS", "ECDFS", "EDFS_a", "EDFS_b", "ELAISS1", "XMM_LSS"]
    RA = [
        150.0925611434997,
        53.15681804806202,
        59.42829027800171,
        63.13103069845953,
        9.481137271096197,
        35.704990247346316,
    ]
    Dec = [
        2.1945447391964454,
        -28.09929786400517,
        -49.13844643312827,
        -47.775308526672944,
        -44.017183266469296,
        -4.754416031251561,
    ]

    df = pd.DataFrame(fields, columns=["field"])
    df["RA"] = RA
    df["DEC"] = Dec
    return df


def chunks_data_split(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def create_standard_catalog(
    query=None,
    source_id_list=None,
    dl_threshold=5000,
):
    if (query is None) & (source_id_list is None):
        raise ValueError("Please give either a query or a source_id_list")
    if (query is not None) & (source_id_list is not None):
        gaia_source_table = []
        for source_id in source_id_list:
            query_id = query + f"  WHERE source_id = {source_id}"
            job = Gaia.launch_job_async(query_id)
            gaia_source_table.append(job.get_results())
        gaia_source_table = vstack(gaia_source_table)
    else:
        job = Gaia.launch_job_async(query)
        gaia_source_table = job.get_results()
    ids = gaia_source_table["source_id"]
    ids_chunks = list(chunks_data_split(ids, dl_threshold))
    datalink_all = []

    retrieval_type = "XP_CONTINUOUS"
    data_structure = "RAW"
    data_release = "Gaia DR3"

    ii = 0
    for chunk in ids_chunks:
        ii = ii + 1
        print(f"Downloading Chunk #{ii}; N_files = {len(chunk)}")
        datalink = Gaia.load_data(
            ids=chunk,
            data_release=data_release,
            retrieval_type=retrieval_type,
            data_structure=data_structure,
            verbose=True,
            output_file=None,
            format="csv",
        )

        datalink_all.append(datalink)

    product_list = [
        item.group_by("source_id")
        for sublist in datalink_all
        for item in sublist[list(datalink.keys())[0]]
    ]
    product_list_groups = [group for item in product_list for group in item.groups]

    gaia_spectra_table = vstack(product_list_groups)

    return gaia_source_table, gaia_spectra_table


def create_standard_catalog_around_fields(
    ra_centers,
    dec_centers,
    delta_ra,
    delta_dec,
    mag_max,
    name,
    star_type_selection=None,
):
    results_fields = []
    for i in range(len(ra_centers)):
        ra_center = ra_centers[i]
        dec_center = dec_centers[i]

        query = f"SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, l, b, phot_variable_flag, ref_epoch \
                FROM gaiadr3.gaia_source \
                WHERE has_xp_sampled = 'True'\
                AND phot_variable_flag != 'VARIABLE'\
                AND astrometric_excess_noise < 0.5 \
                AND ra > {ra_center  - delta_ra} AND ra < {ra_center + delta_ra} \
                AND dec > {dec_center  - delta_dec} AND dec < {dec_center + delta_dec} \
                AND phot_g_mean_mag < {mag_max}"

        job = Gaia.launch_job_async(query)
        results = job.get_results()
        results_fields.append(results)

    results_fields = np.concatenate(results_fields)

    source_id_list = results_fields["source_id"]

    type_stars = np.array(
        get_spType([f"Gaia DR3 {source_id}" for source_id in source_id_list])["spType"]
    )
    if star_type_selection is not None:
        mask_type = np.array([False for _ in range(len(type_stars))])
        for _, typ in enumerate(star_type_selection):
            mask_type |= np.array(
                [type_stars[j][0] == typ for j in range(len(type_stars))]
            )
        source_id_list = source_id_list[mask_type]
        type_stars = type_stars[mask_type]

    query = f"SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, l, b, phot_variable_flag, ref_epoch, rv_template_teff, teff_gspphot \
        FROM gaiadr3.gaia_source"

    gaia_source_table, gaia_spectra_table = create_standard_catalog(
        query=query,
        source_id_list=source_id_list,
        dl_threshold=5000,
    )

    gaia_source_table.to_pandas().to_parquet(f"{name}_source.parquet")
    gaia_spectra_table.to_pandas().to_parquet(f"{name}_spectra.parquet")

    return gaia_source_table, gaia_spectra_table, type_stars


# Catalog selection function of Damiano:


def catalog_selection(
    df,
    sel_variable_flag=None,
    sel_ruwe=False,
    sel_c_star=None,
    sel_proc_mode=None,
    sel_n_transit=None,
    sel_percentage_blended_transit=None,
    sel_percentage_contaminated_transit=None,
    drop_used_flag_columns=True,
):
    """select photometric catalog according to Gaia Flags
    inputs
    ---------
    df : pandas dataframe containg the photometric catalog the files are in /sps/ztf/data/calibrator/gaiaspectra/Photometry/

    sel_variable_flag: string (CONSTANT or VARIABLE) --- select the catalog according to variability flag of Gaia
    (NB in gaia the flags CONSTANT or VARIABLE, NOT AVAILABLE, the function assume the NOT AVAILABLE objects are constant)

    sel_ruwe: bool ---- selection according to ruwe flag (ruwe<1.4) clean a sample from cases
    showing photocentric motions due to unresolved objects (see Riello et al 2021, Montegriffo et al 2022)

    sel_c_star: float --- selection according to color_execess_factor C* defined in Riello et al 2021,
    select star with C*< sel_c_star * sigmaC* (sigmaC* std of C* distribution)

    sel_proc_mod: int (0,1,2) --- select sample according to photometry processing mode: 0 gold sample,
    1 silver, 2 bronze

    sel_n_transit: int --- select sample according total number of transit done to generate total Xp spectra
    (xp_transit = rp_transit + bp_transit)

    sel_percentage_blended_transit: float [0,1] --- select sample according to beta parameter defined in Riello et al 2021
    (beta= (bp_n_blended_transit + rp_n_blended_transit)/ xp_transit

    sel_percentage_contaminated transit: same as blended flag

    drop_used_flag_columns: bool --- chose if you want to remove the flag column in the dataframe after the selection


    see gaia_source or xp_summary files in gaia documentation for more info
    ----------
    return dataframe after selection"""

    used_flag_columns_name = []

    if sel_ruwe:
        df = df[df["ruwe"] < 1.4]
        used_flag_columns_name.append("ruwe")

    if sel_c_star is not None:
        used_flag_columns_name.append("c_star")
        if isinstance(sel_c_star, float):
            df = df[
                df["c_star"]
                < sel_c_star
                * (0.0059898 + 8.817481e-12 * np.power(df["phot_g_mean_mag"], 7.618399))
            ]
        else:
            raise ValueError("sel_c_star should be a float")

    if sel_variable_flag is not None:
        used_flag_columns_name.append("phot_variable_flag")
        if sel_variable_flag == "VARIABLE":
            df = df[df["phot_variable_flag"] == "VARIABLE"]
        elif sel_variable_flag == "CONSTANT":
            df = df[df["phot_variable_flag"] != "VARIABLE"]
        else:
            raise ValueError("accepted variable flag are: VARIABLE and CONSTANT")

    if sel_proc_mode is not None:
        if sel_proc_mode in [0, 1, 2]:
            df = df[df["phot_proc_mode"] == sel_proc_mode]
            used_flag_columns_name.append("phot_proc_mode")
        else:
            raise ValueError("accepted proc_mode are: 0, 1, 2")

    if sel_n_transit is not None:
        if isinstance(sel_n_transit, int):
            df["xp_n_transits"] = df.rp_n_transits + df.bp_n_transits
            df = df[df["xp_n_transits"] > sel_n_transit]
            used_flag_columns_name.extend(
                ["rp_n_transits", "bp_n_transits", "xp_n_transits"]
            )
        else:
            raise ValueError("sel_n_transit should be an integer")

    if sel_percentage_blended_transit is not None:
        if 0.0 <= sel_percentage_blended_transit <= 1.0:
            df["xp_percentage_blended_transits"] = (
                df.rp_n_blended_transits + df.bp_n_blended_transits
            ) / (df.rp_n_transits + df.bp_n_transits)
            df = df[
                df["xp_percentage_blended_transits"]
                < sel_percentage_contaminated_transit
            ]
            used_flag_columns_name.extend(
                [
                    "rp_n_blended_transits",
                    "bp_n_blended_transits",
                    "xp_percentage_blended_transits",
                ]
            )
        else:
            raise ValueError(
                "sel_percentage_blended_transit should be a number between 0 and 1"
            )

    if sel_percentage_contaminated_transit is not None:
        if 0.0 <= sel_percentage_contaminated_transit <= 1.0:
            df["xp_percentage_contaminated_transits"] = (
                df.rp_n_contaminated_transits + df.bp_n_contaminated_transits
            ) / (df.rp_n_transits + df.bp_n_transits)
            df = df[
                df["xp_percentage_contaminated_transits"]
                < sel_percentage_contaminated_transit
            ]
            used_flag_columns_name.extend(
                [
                    "rp_n_contaminated_transits",
                    "bp_n_contaminated_transits",
                    "xp_percentage_contaminated_transits",
                ]
            )
        else:
            raise ValueError(
                "sel_percentage_contaminated_transit should be a number between 0 and 1"
            )

    if drop_used_flag_columns:
        print("removed columns :", used_flag_columns_name)
        df = df.drop(columns=used_flag_columns_name)

    return df


def write_tables(path, gaia_source_table, gaia_spectra_table):
    gaia_source_table.to_pandas().to_parquet(
        os.path.join(path, "gaia_source_file.parquet")
    )
    gaia_spectra_table.to_pandas().to_parquet(
        os.path.join(path, "gaia_spectra_file.parquet")
    )
