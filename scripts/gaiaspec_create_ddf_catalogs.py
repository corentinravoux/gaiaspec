#!/usr/bin/env python3
"""Script: build the Gaia standard-star catalogs around the LSST DDF fields.

Logs in to the Gaia archive and queries bright A/G standard stars within a box
around each Deep Drilling Field centre, downloading their XP spectra and writing
the source / spectra parquet tables. Edit the magnitude / type / box settings at
the top, then run.
"""

from gaiaspec import standard

mag_max_selection = 8.0
class_prefered = ["A", "G"]
delta_ra = 3.0
delta_dec = 3.0

ddf_fields = standard.get_ddf()
ra_centers = ddf_fields["RA"]
dec_centers = ddf_fields["DEC"]

nameout = "gaia_ddf_catalogs"

standard.login()

gaia_source_table, gaia_spectra_table, type_stars = (
    standard.create_standard_catalog_around_fields(
        ra_centers,
        dec_centers,
        delta_ra,
        delta_dec,
        mag_max_selection,
        nameout,
        star_type_selection=class_prefered,
    )
)
