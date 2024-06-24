from astropy.io import ascii
from astropy.table import Table, vstack
from astroquery.gaia import Gaia


def login():
    Gaia.login()


def chunks_data_split(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def create_standard_catalog(
    query,
    dl_threshold=5000,
):

    job = Gaia.launch_job_async(query)
    gaia_source_table = job.get_results()

    ids = gaia_source_table["SOURCE_ID"]
    ids_chunks = list(chunks_data_split(ids, dl_threshold))
    datalink_all = []

    retrieval_type = "XP_CONTINUOUS"
    data_structure = "COMBINED"
    data_release = "Gaia DR3"
    dl_key = f"{retrieval_type}_{data_structure}.csv"

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
        for item in sublist[dl_key]
    ]
    product_list_groups = [group for item in product_list for group in item.groups]

    gaia_spectra_table = vstack(product_list_groups)

    return gaia_source_table, gaia_spectra_table
