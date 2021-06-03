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

spectral_path = 'catalogs/'
spectral_names = ['alldeep.egs.uniq.2012jun13']
matched_path = '/n17data/jimenez/shaperun/ps3pi_cfis/deep23/output/run_sp_PcM_2021-03-01_19-30-24/paste_cat_runner/output/'
unmatched_path = '/n17data/jimenez/shaperun/ps3pi_cfis/deep23/output/run_sp_PcD_2021-03-01_18-08-31/paste_cat_runner/output/'
spectral_surveys = ['deep23']
vignet = False

#------------------------------------------------------------------#
# # # # # MACHINE LEARNING PARAMS # # # # #
#------------------------------------------------------------------#

max_evals = 50
path_to_csv = None
#weights = path + 'output/ps3pi_cfis/'+output_name+'/files/' + 'Weights_' + output_name + '.npy'
weights = True
cv = 4
feature_engineering = True
feature_importance = False
plot = True
morphological_parameters = False
morph_importance = False
#path_to_csv = path + 'catalogs/' + 'MediumDeep_CFHT_CFIS_R_matched_catalog_2' + '.csv'



