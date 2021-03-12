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
#temp_path = '/n17data/jimenez/temp/'
temp_path = path

#------------------------------------------------------------------#
# # # # # PREPROCESSING PARAMS # # # # #
#------------------------------------------------------------------#

spectral_path = '/n17data/jimenez/spectral_catalogs/'
spectral_names = ['alldeep.egs.uniq.2012jun13']
path_to_tile_run = '/n17data/jimenez/shaperun/'
input_path = None
spectral_surveys = ['unmatched_deep23']
vignet = False

#------------------------------------------------------------------#
# # # # # MACHINE LEARNING PARAMS # # # # #
#------------------------------------------------------------------#

max_evals = 100
path_to_csv = None
#weights = path + 'output/ps3pi_cfis/files/' + 'Weights_' + output_name + '.npy'
weights = False
cv = 20
feature_engineering = False
feature_importance = False
plot = False
morph_importance = True
#path_to_csv = path + 'catalogs/' + 'MediumDeep_CFHT_CFIS_R_matched_catalog_2' + '.csv'



