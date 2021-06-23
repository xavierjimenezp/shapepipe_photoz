#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:02:58 2020

@author: Xavier Jimenez
"""

#------------------------------------------------------------------#
# # # # # PATH # # # # #
#------------------------------------------------------------------#

import os
path = os.getcwd() + '/'

#------------------------------------------------------------------#
# # # # # GLOBAL PARAMS # # # # #
#------------------------------------------------------------------#

bands = ['R']
output_path = path
output_name = 'CFIS_matched_deep_2_3_catalog_R'
temp_path = path + 'temp/'

#------------------------------------------------------------------#
# # # # # PREPROCESSING PARAMS # # # # #
#------------------------------------------------------------------#

catalog_path = 'catalogs/'
spectral_path = catalog_path
spectral_names = ['alldeep.egs.uniq.2012jun13']
matched_path = f'{catalog_path}/matched/'
unmatched_path = f'{catalog_path}/unmatched/'
spectral_surveys = ['deep23']
vignet = False

#------------------------------------------------------------------#
# # # # # MACHINE LEARNING PARAMS # # # # #
#------------------------------------------------------------------#

max_evals = 50
path_to_csv = None
weights = True
cv = 4
feature_engineering = True
feature_importance = False
plot = True
morphological_parameters = False
morph_importance = False
#path_to_csv = path + 'catalogs/' + 'MediumDeep_CFHT_CFIS_R_matched_catalog_2' + '.csv'



