multi_band_matched
==================

Multi-band UNIONS catalogues matched to
spectroscopic surveys.

W3_deep23
---------

CFHTLenS-W3 field.

CFIS-r, CFHTLenS-u, Pan-STARRS Medium-Deep-i, z
DEEP2+3.

Files
^^^^^

	- CFIS_matched_deep_2_3_catalog_R.csv
	  CFIS-r + morphology, matched with DEEP23
	- CFIS_matched_deep_2_3_catalog_R_preprocessed.csv
	  CFIS-r + DEEP23-z_spectro
	- MediumDeep_IZG_CFHT_U_CFIS_R_catalog_unmatched.csv
	  Magnitudes r, u, i, z, g
	- CFIS_matched_deep_2_3_catalog_RIZGY.csv
	  CFIS CFHTlenS DEEP23 matched catalogue (?)
        - R_CFHT_vs_CFIS.csv
          CFIS r-band magnitudes matched with CFHTLenS

W3_deep23/ps3pi_cfis
--------------------

Catalogues required to run photo.pz from
https://github.com/xavierjimenezp/shapepipe_photoz
to compute photometric redshifts.

Files
^^^^^
	- alldeep.egs.uniq.2012jun13.fits
	  Matched catalogue with DEEP2+3
	- matched/cat*
	  ShapePipe output catalogues matched with DEEP2+3
	- unmatched/cat*
	  ShapePipe output catalogues not matched with DEEP2+3,
	  to compute weights, to approximate input target photometric
	  catalogue with matched selection.
