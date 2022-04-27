
# coding: utf-8

# # Matched UNIONS 800 deg^2 + ShapePipe shear catalogue
# Need to join using CFIS_ID.

# In[ ]:


import os
import numpy as np
import math

import pandas as pd
import matplotlib.pylab as plt

from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as units

from collections import Counter

from functions import *

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read matched ShapePipe shear catalogue

# In[ ]:


base_name_sp = 'shape_catalog_extended_ngmix'

hdu = fits.open(f'sp_output/{base_name_sp}.fits')

data = hdu[1].data


# In[ ]:


# Print column names
data_keys = data.dtype.names
print(data_keys)


# ## Plots

# ### Spatial distribution

# In[ ]:


plt.plot(data['RA'], data['Dec'], '.')
plt.title('UNIONS-800 deg$^2$ + ShapePipe galaxies')
plt.xlabel('RA [deg]')
_ = plt.ylabel('DEC [deg]')


# In[ ]:


def millify(n):
    """Millify

    Return human-readible names of large numbers

    Parameters
    ----------
    n : int
        input number

    Returns
    -------
    s : string
        output name
    """

    millnames = ['',' Thousand',' Million',' Billion',' Trillion']

    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        )
    )

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


# In[ ]:


ngal = len(data)
print(f'Number of galaxies = {ngal} = {millify(ngal)}')


# ### Magnitude distribution (r-band)

# In[ ]:


fig, (ax) = plt.subplots(1, 1)

range = (19.8, 25.5)

h = ax.hist(data['mag'], bins=100, density=True, range=range)
plt.title('UNIONS-800 deg$^2$ + SP')
plt.xlabel('$r$ (ShapePipe)')
plt.xticks(ticks=[20, 21, 22, 23, 24, 25])
_ = plt.ylabel('frequency')


# ### Read multi-band UNIONS catalogue

# In[ ]:


# UNIONS

data_dir = f'{os.environ["HOME"]}/astro/data/CFIS/800deg2'

# Full catalogue
base_name_ph = 'unions.ugriz2'

# Smaller catalogue
#base_name_ph = 'unions.ugriz_1M'

cat_ph = pd.read_csv(f'{data_dir}/{base_name_ph}.tsv', delimiter='\s+')


# In[ ]:


print('Keys in UNIONS catalogue:')
mb_keys = cat_ph.keys()
print(mb_keys)
print(f'Number of UNIONS objects = {len(cat_ph)}')


# ### Match the two catalogues acording to the CFIS galaxy ID

# In[ ]:


# Transform IDs back to int
cfis_id = data['CFIS_ID'].astype(int)


# In[ ]:


# Get indices of matches
xy, x_ind, y_ind = np.intersect1d(cfis_id, cat_ph['CFIS_ID'], return_indices=True)


# In[ ]:


# Check
print(cfis_id[x_ind][:10])
print(cat_ph['CFIS_ID'][y_ind][:10])


# In[ ]:


len(cat_ph['CFIS_ID'][y_ind])


# In[ ]:


len(data)


# In[ ]:


# Create column array for SP catalogue
cols = []
for key in data_keys:
    print(key)
    if key != 'CFIS_ID':
        c = fits.Column(name=key, format='D', array=data[x_ind][key])
    else:
        dat = cfis_id
        c = fits.Column(name=key, format='I64', array=dat[x_ind])
    cols.append(c)


# In[ ]:


# Append columns from multi-band catalogue
for idx, key in enumerate(mb_keys):
    if key in ['RA', 'Dec']:
        # Coordinates in the matched MB+SP catalogue are exactly the same,
        # so we don't need them again
        continue
    elif key == 'CFIS_ID':
        # Add mb CFIS_ID for consistency check later
        key2 = f'{key}_mb'
    else:
        key2 = key
    print(key2)
    c = fits.Column(name=key2, format=cat_ph.dtypes[idx], array=cat_ph[key][y_ind])
    cols.append(c)


# In[ ]:


out_name = 'cfis_800d2_SP_v1.0.fits'
hdu_out = fits.BinTableHDU.from_columns(cols)
hdu_out.writeto(out_name)


# In[ ]:


print('Saved FITS file f{out_name}')

