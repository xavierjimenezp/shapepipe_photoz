#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:02:58 2020

@author: Xavier Jimenez
"""

#------------------------------------------------------------------#
# # # # # Imports # # # # #
#------------------------------------------------------------------#

import numpy as np
import os
import shutil
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from sklearn import kernel_ridge
from sklearn.utils import validation
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
from matplotlib.colors import LogNorm

import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.stats import mad_std


from joblib import Parallel, delayed
from tqdm import tqdm
import argparse

import tensorflow as tf
import tensorflow.keras as tfk
import keras
from tensorflow.keras import regularizers

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from scipy.stats import norm
from scipy import stats


import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

# import shap
import eli5
from eli5.sklearn import PermutationImportance

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope
from hyperas import optim
from hyperas.distributions import choice, uniform

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR


#------------------------------------------------------------------#
# # # # # Functions # # # # #
#------------------------------------------------------------------#


class GenerateFiles(object):
    """[summary]
    """

    def __init__(self, survey, bands, temp_path):
        """[summary]

        Args:
            survey ([type]): [description]
            bands ([type]): [description]
            temp_path ([type]): [description]
        """

        self._path = os.getcwd() +'/'
        self.survey = survey
        self._bands = bands
        self.temp_path = temp_path


    def make_directory(self, path_to_file):
        """[summary]

        Args:
            filename ([type]): [description]
        """

        try:
            os.mkdir(path_to_file)
        except OSError:
            pass
        else:
            print ("Successfully created the directory %s " % path_to_file)


    def make_directories(self):
        """[summary]
        """

        self.make_directory(self.temp_path + self.survey)

        temp_directories = ['matched', 'unmatched', 'vignet', 'spectral_surveys', 'd2d', 'redshift']

        for temp_name in temp_directories:
            self.make_directory(self.temp_path + self.survey + '/' + temp_name)

        for band in self._bands:
            self.make_directory(self.temp_path + self.survey + '/vignet/' + band)

        self.make_directory(self.temp_path + self.survey + '/vignet/array')

        self.make_directory(self._path + 'output')
        self.make_directory(self._path + 'output/' + self.survey)
        self.make_directory(self._path + 'output/' + self.survey + '/files')
        self.make_directory(self._path + 'output/' + self.survey + '/files/ML')
        self.make_directory(self._path + 'output/' + self.survey + '/figures')




    def remove_files_from_directory(self, folder):
        """Removes files for a given directory

        Args:
            folder ([str]): directory path
        """

        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


    def clean_temp_directories(self):
        """[summary]
        """

        if os.path.exists(self.temp_path + self.survey) and os.path.isdir(self.temp_path + self.survey):
            if not os.listdir(self.temp_path + self.survey):
                print("Directory %s is empty"%(self.temp_path + self.survey))
            else:
                temp_directories = ['/d2d/', '/matched/',  '/unmatched/', '/vignet/', '/redshift/', '/spectral_surveys/']
                for directory in temp_directories:
                    self.remove_files_from_directory(self.temp_path + self.survey + directory)
                    print("Successfully removed the directory %s " % (self.temp_path + self.survey + directory))
        else:
            print("Directory %s does not exists"%(self.temp_path + self.survey))


    def is_directory_empty(self, path_to_dir):
        """Checks if given directory is empty or not.

        Args:
            path_to_dir ([str]): path to the directory

        Returns:
            [bool]: True if directory is empty, False if not.
        """

        if os.path.exists(path_to_dir) and os.path.isdir(path_to_dir):
            if not os.listdir(path_to_dir):
                print("Directory %s is empty"%path_to_dir)
                return True
            else:
                print("Directory %s is not empty"%path_to_dir)
                return False
        else:
            print("Directory %s don't exists"%path_to_dir)

class GeneratePlots(object):

    def __init__(self, survey, bands, temp_path, csv_name, spectral_names, output_name='match_cat'):

        self._path = os.getcwd() + '/'
        self.survey = survey
        self._bands = bands
        self.temp_path = temp_path
        self.csv_name = csv_name
        self.output_name = output_name
        self.spectral_names = spectral_names
        self.output_path = self._path + 'output/' + self.survey + '/figures'

        self.df_matched = pd.read_csv(self._path + 'output/' + self.survey + '/files/' + self.csv_name + '.csv')
        self.df_unmatched = pd.read_csv(self._path + 'output/' + self.survey + '/files/' + self.csv_name + '_unmatched'+'.csv')
        # self.df_d2d = pd.read_csv(self._path + 'output/' + self.survey + '/files/' + self.csv_name +'_d2d'+'.csv')


    def plot_d2d(self):
        """
        Distance distribution plot for matched galaxies
        """

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_xlabel(r'$\mathrm{angular\;distance\;\left(arcsec\right)}$', fontsize=20)
        ax.set_ylabel(r'$\mathrm{matched\;galaxies\;in}$'+' band R', fontsize=20)

        ax.hist(np.array(self.df_d2d['d2d'].values)*3600, bins = 50)

        plt.savefig(self.output_path + self.output_name + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()


    def plot_matched_z_spec_hist(self):
        """
        z_spec hist for matched galaxies
        """

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_xlabel(r'$\mathrm{spectral\;redshift}$', fontsize=20)
        ax.set_ylabel(r'$\mathrm{matched\;galaxies\;in}$'+' band R', fontsize=20)

        ax.axvline(np.median(self.df_matched['Z_SPEC'].values), ls='--', color='k', label='median: {:.2f}'.format(np.median(self.df_matched['Z_SPEC'].values)))
        ax.hist(self.df_matched['Z_SPEC'].values, bins = 50)

        legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
        legend.get_frame().set_facecolor('C0')

        plt.savefig(self.output_path + self.output_name + '_z_hist_' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()


    def plot_matched_r_i_i_z(self):
        """
        R-I vs I-Z 2d histogram plot for matched galaxies
        """

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_ylabel(r'$\mathrm{R-I}$', fontsize=20)
        ax.set_xlabel(r'$\mathrm{I-Z}$', fontsize=20)

        im = ax.hist2d((self.df_matched['MAG_AUTO_I'] - self.df_matched['MAG_AUTO_Z']).values, (self.df_matched['MAG_AUTO_R'] - self.df_matched['MAG_AUTO_I']).values, bins=(50,50), cmap='Blues')
        cbar = fig.colorbar(im[3])
        ax.set_xlim([-5, 5])
        ax.set_ylim([-2, 4])

        plt.suptitle(r'$\mathrm{R-I\; vs \;I-Z \;for\; matched\; galaxies}$', fontsize=20, y=1.0)
        plt.savefig(self.output_path + 'R-I_I-Z' + '_matched' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()


    def plot_matched_u_r_r_i(self):
        """
        U-R vs R-I 2d histogram plot for matched galaxies
        """

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_ylabel(r'$\mathrm{U-R}$', fontsize=20)
        ax.set_xlabel(r'$\mathrm{R-I}$', fontsize=20)

        im = ax.hist2d((self.df_matched['MAG_AUTO_R'] - self.df_matched['MAG_AUTO_I']).values, (self.df_matched['MAG_AUTO_U'] - self.df_matched['MAG_AUTO_R']).values, bins=(50,50), cmap='Blues')
        cbar = fig.colorbar(im[3])
        ax.set_xlim([-2, 4])
        ax.set_ylim([0, 5])

        plt.suptitle(r'$\mathrm{U-R\; vs \;R-I \;for\; matched\; galaxies}$', fontsize=20, y=1.0)
        plt.savefig(self.output_path + 'U-R_R-I' + '_matched' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()


    def plot_unmatched_r_i_i_z(self):
        """
        R-I vs I-Z 2d histogram plot for all galaxies
        """

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_ylabel(r'$\mathrm{R-I}$', fontsize=20)
        ax.set_xlabel(r'$\mathrm{I-Z}$', fontsize=20)

        im = ax.hist2d((self.df_unmatched['MAG_AUTO_I'] - self.df_unmatched['MAG_AUTO_Z']).values, (self.df_unmatched['MAG_AUTO_R'] - self.df_unmatched['MAG_AUTO_I']).values, bins=(50,50), cmap='Blues')
        cbar = fig.colorbar(im[3])
        cbar.formatter.set_powerlimits((0, 0))
        ax.set_xlim([-5, 5])
        ax.set_ylim([-2, 4])

        plt.suptitle(r'$\mathrm{R-I\; vs \;I-Z \;for\; all\; galaxies}$', fontsize=20, y=1.0)
        plt.savefig(self.output_path + 'R-I_I-Z' + '_unmatched' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()


    def plot_unmatched_u_r_r_i(self):
        """
        U-R vs R-I 2d histogram plot for all galaxies
        """

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_ylabel(r'$\mathrm{U-R}$', fontsize=20)
        ax.set_xlabel(r'$\mathrm{R-I}$', fontsize=20)

        im = ax.hist2d((self.df_unmatched['MAG_AUTO_R'] - self.df_unmatched['MAG_AUTO_I']).values, (self.df_unmatched['MAG_AUTO_U'] - self.df_unmatched['MAG_AUTO_R']).values, bins=(50,50), cmap='Blues')
        cbar = fig.colorbar(im[3])
        cbar.formatter.set_powerlimits((0, 0))
        ax.set_xlim([-2, 4])
        ax.set_ylim([0, 5])

        plt.suptitle(r'$\mathrm{U-R\; vs \;R-I \;for\; all\; galaxies}$', fontsize=20, y=1.0)
        plt.savefig(self.output_path + 'U-R_R-I' + '_unmatched' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()


    def plot_unmatched_z_spec_hist(self):
        """
        Z unmatched distribution plot
        """

        df_Z = pd.DataFrame(data={'RA': [], 'DEC': [], 'Z_SPEC': []})
        for spectral_name in self.spectral_names:
            df_Z = pd.concat((df_Z, pd.read_csv(self.temp_path + self.survey + '/spectral_surveys/'+'z_%s'%spectral_name + '.csv')))

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_xlabel(r'$\mathrm{spectral\;redshift}$', fontsize=20)
        ax.set_ylabel(r'$\mathrm{all\;spectral\;galaxies}$', fontsize=20)

        ax.axvline(np.median(df_Z['Z_SPEC'].values), ls='--', color='k', label='median: {:.2f}'.format(np.median(df_Z['Z_SPEC'].values)))
        ax.hist(df_Z['Z_SPEC'].values, bins = 50)

        legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
        legend.get_frame().set_facecolor('C0')


        plt.savefig(self.output_path + 'cat_' + '_z_hist_' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()



class MakeCatalogs(object):

    def __init__(self, survey, bands, temp_path):
        """[summary]

        Args:
            survey ([type]): [description]
            bands ([type]): [description]
            temp_path ([type]): [description]
        """

        self._path = os.getcwd() + '/'
        self.survey = survey
        self._bands = bands
        self.temp_path = temp_path


    def make_survey_catalog(self, spectral_path, spectral_name):
        """[summary]

        Args:
            spectral_path ([type]): [description]
        """

        # df = pd.DataFrame(data={'RA': [], 'DEC': [], 'Z_SPEC': []})
        # for spectral_name in spectral_names:
        HDU = fits.open(spectral_path + '/' + spectral_name + '.fits')
        RA = HDU[1].data['ra']
        DEC = HDU[1].data['dec']
        Z_SPEC = HDU[1].data['z']
        df = pd.DataFrame(data={'RA': RA.tolist(), 'DEC': DEC.tolist(), 'Z_SPEC': Z_SPEC.tolist()})

        # Keep galaxies with positive redshift only
        df.query("Z_SPEC > 0", inplace=True)

        # Check for duplicate coordinates
        pcatalog = SkyCoord(ra=df['RA'].values, dec=df['DEC'].values, unit='deg')
        isdup = self.check_dup_coord(coords=pcatalog)
        df['isdup'] = isdup
        df.query("isdup == False", inplace=True)
        df.drop(columns=['isdup'], inplace=True)

        df.to_csv(self.temp_path + self.survey + '/spectral_surveys/'+'z_%s'%spectral_name+'.csv', index=False)



    def check_dup_coord(self, coords, tol=0.3*u.arcsecond):
        """ Checks for duplicates in a list or array
        of coordinates, within an angular distance
        tolerance 'tol'.

        Parameters
        ----------
        coords : SkyCoord array or list
            Coordinates to check for duplicates
        tol : Angle, optional
            Angular tolerance

        Returns
        -------
        isdup : boolean array, shape(len(coords),)
            Whether a given coordinate in `coords` is a duplicate
        """

        _, d2d, _ = match_coordinates_sky(coords, coords, nthneighbor=2)
        isdup = d2d < tol

        return isdup


    def make_catalog(self, p, paste_dir, input_path, spectral_name, vignet=True):
        """Creates .csv catalogs with different flux & magnitudes in bands R,I,Z,G,Y
        as well as morphological parameters which are matched with a spectral catalog.
        Creates .npy 4D ndarrays & .png images for vignets

        Args:
            p ([int]): Tile id (0 =< id < len(paste_dir))
            paste_dir ([list]): list containing tile names
            input_path ([str]): directory path for paste_dir
            spectral_name ([str]): [description]
            vignet (bool, optional): [description]. Defaults to True.
        """

        #------------------------------------------------------------------#
        # # # # # Extract data from catalogs # # # # #
        #------------------------------------------------------------------#

        df_Z = pd.read_csv(self.temp_path + self.survey + '/spectral_surveys/'+'z_%s'%spectral_name + '.csv')

        file_name = paste_dir[p]
        HDU_tile = fits.open(input_path + file_name)

        # BANDS = dict(zip(self._bands, np.arange(2,len(self._bands)+2)))
        # BANDS = {'R':2, 'I':3, 'Z':4, 'G':5, 'Y':6}

        band = 'R'

        RA = HDU_tile[band].data['X_WORLD'].tolist()
        DEC = HDU_tile[band].data['Y_WORLD'].tolist()

        SNR = HDU_tile[band].data['SNR_WIN'].tolist()
        FWHM = HDU_tile[band].data['FWHM_WORLD'].tolist()
        ELONGATION = HDU_tile[band].data['ELONGATION'].tolist()

        MAG = HDU_tile[band].data['MAG_AUTO'].tolist()
        MAG_ERR = HDU_tile[band].data['MAGERR_AUTO'].tolist()

        ID = np.arange(0, len(HDU_tile[band].data['X_WORLD'].tolist()), 1)

        df_to_cut = pd.DataFrame(data={'ID': ID, 'RA': RA, 'DEC': DEC, 'SNR': SNR, 'FWHM': FWHM, 'MAG': MAG, 'MAG_ERR': MAG_ERR, 'ELONGATION': ELONGATION})

        #------------------------------------------------------------------#
        # # # # # SNR & FWHM & MAG & MAG_ERR CUTS  # # # # #
        #------------------------------------------------------------------#

        if self.survey == 'unions':
            df_to_cut.query("SNR > 10 & FWHM*3600 > 0.8 & MAG < 30 & MAG_ERR < 0.5", inplace=True)
        if self.survey == 'ps3pi_cfis':
            df_to_cut.query("SNR > 10 & SNR < 500 & FWHM*3600 > 0.8 & MAG < 30 & MAG_ERR < 0.5", inplace=True)
        df_to_cut.drop(columns=['SNR', 'MAG_ERR', 'MAG'], inplace=True)

        #------------------------------------------------------------------#
        # # # # # Check for duplicate coordinates # # # # #
        #------------------------------------------------------------------#

        scatalog = SkyCoord(ra=df_to_cut['RA'].values, dec=df_to_cut['DEC'].values, unit='deg')
        isdup = self.check_dup_coord(coords=scatalog)

        # sub arrays without duplicates for both catalogs
        df_to_cut['isdup'] = isdup
        df_to_cut.query("isdup == False", inplace=True)
        df_to_cut.drop(columns=['isdup'], inplace=True)

        #------------------------------------------------------------------#
        # # # # # Spatial coordinates matching # # # # #
        #------------------------------------------------------------------#

        scatalog_sub = SkyCoord(ra=df_to_cut['RA'].values, dec=df_to_cut['DEC'].values, unit='deg')
        pcatalog_sub = SkyCoord(ra=df_Z['RA'].values, dec=df_Z['DEC'].values, unit='deg')
        idx, d2d, _ = match_coordinates_sky(scatalog_sub, pcatalog_sub, nthneighbor=1)

        tol = 0.3*u.arcsecond #threshold to consider whether or not two galaxies are the same
        ismatched = d2d < tol

        # try:

        #------------------------------------------------------------------#
        # # # # # Create matched dataframe with redshift # # # # #
        #------------------------------------------------------------------#

        df_d2d = pd.DataFrame(data={'ismatched': ismatched, 'idx': idx, 'd2d': d2d, 'RA': df_to_cut['RA'].values,'DEC':df_to_cut['DEC'].values})
        df_d2d.query("ismatched == True", inplace=True)
        df_d2d.drop(columns=['ismatched'], inplace=True)

        if self.survey == 'unions':
            try:
                idx_sub = np.array([(i,ide) for (i,ide) in enumerate(idx) if ismatched[i] == True])[:,1]
            except:
                print('0 matched objects wo filters for tile %s'%(file_name[:-5]+'_'+spectral_name))
                return

            z_spec_sub = []
            for i in idx_sub:
                z_spec_sub.append(df_Z['Z_SPEC'].values[i])
            z_spec_sub = np.array(z_spec_sub)

            df_to_cut['ismatched'] = ismatched
            df_to_cut.query("ismatched == True", inplace=True)
            df_to_cut.drop(columns=['ismatched'], inplace=True)
            df_to_cut['Z_SPEC'] = z_spec_sub

        #------------------------------------------------------------------#
        # # # # # Band extraction for dataframe # # # # #
        #------------------------------------------------------------------#

        PARAMS = ['FWHM_WORLD', 'SNR_WIN' ,'MAG_AUTO','MAGERR_AUTO', 'MAG_WIN', 'MAGERR_WIN', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_WIN', 'FLUXERR_WIN', 'FLUX_APER', 'FLUXERR_APER']
        NAMES = np.array(['%s_%s'%(param, band) for param in PARAMS for band in self._bands])
        MAG_ZEROPOINT_NAMES = np.array(['MAG_AUTO_%s'%band for band in self._bands[1:]])
        MAG_NAMES = ['MAG_AUTO', 'MAG_WIN']
        MAG_COLS = np.array(['%s_%s'%(param, band) for param in MAG_NAMES for band in self._bands[1:]])



        # try:
        try:
            par = [HDU_tile[band].data[param].tolist() for param in PARAMS for band in self._bands]
        except:
            print('Tile %s does not have the right number of bands'%(file_name[:-5]+'_'+spectral_name))
            return
        df_mag = pd.DataFrame(data = dict(zip(NAMES, par)))
        NAME = [file_name[4:-5]+'_%s'%i for i in range(len(df_mag))]
        ide = np.arange(0, len(df_mag), 1)
        df_mag.insert(0, 'ide', ide)
        df_mag.insert(0, 'NAME', NAME)   



        if self.survey == 'ps3pi_cfis':
            df_mag['Z_SPEC'] = HDU_tile['R'].data['Z'].tolist()
            df_mag['id'] = HDU_tile[self._bands[0]].data['NUMBER'].tolist()
            df_mag['ID'] = np.arange(0,len(df_mag),1).tolist()
            ## merge
            df_mag = pd.merge(df_mag, df_to_cut, indicator=True, on='ID', how='outer').query('_merge=="both"').drop('_merge', axis=1)

            MORPHO_NAMES = ['id', 'gal_g1', 'gal_g1_err', 'gal_g2', 'gal_g2_err', 'gal_gini', 'gal_sb', 'gal_rho4', 'gal_sigma', 'gal_resolution', 'psf_sigma']
            morpho_par = [HDU_tile['R_PSF'].data[param].tolist() for param in MORPHO_NAMES]
            df_morph = pd.DataFrame(data = dict(zip(MORPHO_NAMES, morpho_par)))
            
            dfbands = pd.merge(df_mag, df_morph, indicator=True, on='id', how='outer').query('_merge=="both"').drop('_merge', axis=1)
            dfbands.drop(columns=['id'], inplace=True)
            
            dfbands_matched = dfbands.copy()
            dfbands_unmatched = dfbands.copy()

        if self.survey == 'unions':
            dfbands = df_mag.copy()
            dfbands['ID'] = np.arange(0,len(dfbands),1).tolist()
            ## merge
            dfbands_matched = pd.merge(dfbands, df_to_cut, indicator=True, on='ID', how='left').query('_merge=="both"').drop('_merge', axis=1)
            df_unmatched = df_to_cut.copy()
            dfbands_unmatched = pd.merge(dfbands, df_unmatched, indicator=True, on='ID', how='left').query('_merge=="both"').drop('_merge', axis=1)




        if self.survey == 'ps3pi_cfis':
            print(np.corrcoef(dfbands_matched['FWHM'], dfbands_matched['gal_sigma']))


        # replace 99 by np.nan in MAG cols (TO DO: replace whatever equivalent there is for FLUX cols)
        for col in MAG_COLS:
            dfbands_matched.loc[(dfbands_matched[col]>98), col] = np.nan
            dfbands_unmatched.loc[(dfbands_unmatched[col]>98), col] = np.nan

        for band in self._bands:
            if self.survey == 'ps3pi_cfis':
                dfbands_matched.query("SNR_WIN_%s > 10 & SNR_WIN_%s < 500 & FWHM_WORLD_%s*3600 > 0.8 & MAGERR_AUTO_%s < 0.5" %(band,band,band,band) , inplace=True)
                dfbands_unmatched.query("SNR_WIN_%s > 10 & SNR_WIN_%s < 500 & FWHM_WORLD_%s*3600 > 0.8 & MAGERR_AUTO_%s < 0.5" %(band,band,band,band) , inplace=True)
            else:
                dfbands_matched.query("SNR_WIN_%s > 10 & FWHM_WORLD_%s*3600 > 0.8 & MAGERR_AUTO_%s < 0.5" %(band,band,band) , inplace=True)
                dfbands_unmatched.query("SNR_WIN_%s > 10 & FWHM_WORLD_%s*3600 > 0.8 & MAGERR_AUTO_%s < 0.5" %(band,band,band) , inplace=True)                
        for band in self._bands:
            dfbands_matched.drop(columns=['SNR_WIN_%s'%band, 'FWHM_WORLD_%s'%band], inplace=True)
            dfbands_unmatched.drop(columns=['SNR_WIN_%s'%band, 'FWHM_WORLD_%s'%band], inplace=True)

        # keep only mags with err<0.5
        # for name in MAGERR_NAMES:
            # dfbands_matched.query("%s < 0.5"%name, inplace=True)
            # dfbands_unmatched.query("%s < 0.5"%name, inplace=True)

        # shift -5 mag
        if self.survey == 'ps3pi_cfis':
            for name in MAG_ZEROPOINT_NAMES:
                dfbands_matched[name] = dfbands_matched[name] - 5
                dfbands_unmatched[name] = dfbands_unmatched[name] - 5

        # save png vignets & array
        if vignet == True:
            self.vignet_to_png(file_name, dfbands_matched, HDU_tile)
            self.vignet_to_array(file_name, dfbands_matched, HDU_tile)


        # print(np.corrcoef(dfbands_matched['MAG_AUTO_R'], dfbands_matched['Z_SPEC']))
        print('%i matched objects w filters for tile %s'%(len(dfbands_matched), file_name[:-5]+'_'+spectral_name))

        dfbands_matched.drop(columns=['ID'], inplace=True)
        dfbands_matched.to_csv(self.temp_path + self.survey + '/matched/'+'%s'%file_name[:-5]+'_'+spectral_name+'.csv', index=False)
        dfbands_unmatched.drop(columns=['ID'], inplace=True)
        dfbands_unmatched.to_csv(self.temp_path + self.survey + '/unmatched/'+'%s'%file_name[:-5]+'_'+spectral_name+'.csv', index=False)

        df_Z.to_csv(self.temp_path + self.survey + '/redshift/'+'%s'%file_name[:-5]+'_'+spectral_name+'.csv', index=False)

        # df_d2d.to_csv(self.temp_path + self.survey + '/d2d/'+'%s'%file_name[:-5]+'_'+spectral_name+'.csv', index=False)

        #     except:
        #         print('Tile %s does not have the right number of bands'%(file_name[:-5]+'_'+spectral_name))

        # except:
        #     print('0 matched objects for tile %s'%(file_name[:-5]+'_'+spectral_name))



            # print('Tile %s does not have the right number of bands'%(file_name[:-5]+'_'+spectral_name))
        # except:
        #     print('Tile %s did not match any spectral galaxy'%(file_name[:-5]+'_'+spectral_name))



    def vignet_to_png(self, file_name, dfbands_matched, HDU_tile):
        """[summary]

        Args:
            file_name ([type]): [description]
            dfbands_matched ([type]): [description]
            HDU_tile ([type]): [description]
        """

        # BANDS = dict(zip(self._bands, np.arange(2,len(self._bands)+2)))

        for i,id in enumerate(dfbands_matched['ID']):
            for band in self._bands:
                plt.ioff()
                plt.figure(figsize=(3,3), tight_layout=False)
                plt.imshow(HDU_tile[band].data['VIGNET'][int(id)], norm=LogNorm())
                plt.colorbar()
                plt.title('MAG_{} =  {:.1f}'.format(band, dfbands_matched['MAG_AUTO_%s'%band].values[i]))
                plt.savefig(self.temp_path + self.survey + "/vignet/%s/vignet_%s_%s_%s-.png"%(band, band, file_name[:-5], dfbands_matched['ide'].values[i]), bbox_inches='tight', transparent=True)
                plt.close()
        plt.show()


    def vignet_to_array(self, file_name, dfbands_matched, HDU_tile):
        """[summary]

        Args:
            file_name ([type]): [description]
            dfbands_matched ([type]): [description]
            HDU_tile ([type]): [description]
        """

        X = np.ndarray((len(dfbands_matched),51,51,len(self._bands)))

        for i,id in enumerate(dfbands_matched['ID']):
            for j,band in enumerate(self._bands):
                HDU = HDU_tile[band].data['VIGNET'][int(id)]
                X[i,:,:,j] = np.asarray(np.where(HDU > -1, HDU, 0))

        np.save(self.temp_path + self.survey + '/vignet/array/vignet_%s'% file_name[:-5], X)


    def merge_catalogs(self, output_name, vignet=True):
        """Merges individual tile catalogs

        Args:
            output_name ([type]): [description]
        """

        #------------------------------------------------------------------#
        # # # # # CSV files # # # # #
        #------------------------------------------------------------------#

        # Import temp files
        path_matched = self.temp_path + self.survey + '/matched/'
        all_files = glob.glob(os.path.join(path_matched, "*.csv"))
        df_matched_short = pd.concat((pd.read_csv(f) for f in all_files), axis=0)
        # Remove duplicates
        scatalog = SkyCoord(ra=df_matched_short['RA'].values, dec=df_matched_short['DEC'].values, unit='deg')
        isdup = self.check_dup_coord(coords=scatalog)
        df_matched_short['isdup'] = isdup
        df_matched_short.query("isdup == False", inplace=True)
        df_matched_short.drop(columns=['isdup'], inplace=True)
        # Save file
        df_matched_short.to_csv(self._path + 'output/' + self.survey + '/files/' + output_name + '.csv', index=False)

        if vignet == True:
            # Import temp files
            path_matched_vignet = self.temp_path + self.survey + '/vignet/array/'
            all_files = glob.glob(os.path.join(path_matched_vignet, "*.npy"))
            X = []
            for f in all_files:
                X.append(np.load(f))
            vignet = np.concatenate(X, axis=0)
            # Remove duplicates
            vignet_nodup = np.ndarray((len(df_matched_short),51,51,5))
            k = 0
            for i,dup in enumerate(isdup):
                if dup == False:
                    vignet_nodup[k,:,:,:] = vignet[i,:,:,:]
                    k += 1
            # Save file
            np.save(self._path + 'output/' + self.survey + '/files/' + output_name, vignet_nodup)


        # # Import temp files
        # path_matched_d2d = self.temp_path + self.survey + '/d2d/'
        # all_files = glob.glob(os.path.join(path_matched_d2d, "*.csv"))
        # df_matched_d2d = pd.concat((pd.read_csv(f) for f in all_files), axis=0)
        # # Remove duplicates
        # scatalog = SkyCoord(ra=df_matched_d2d['RA'].values, dec=df_matched_d2d['DEC'].values, unit='deg')
        # isdup = self.check_dup_coord(coords=scatalog)
        # df_matched_d2d['isdup'] = isdup
        # df_matched_d2d.query("isdup == False", inplace=True)
        # df_matched_d2d.drop(columns=['isdup', 'RA', 'DEC'], inplace=True)
        # # Save file
        # df_matched_d2d.to_csv(self._path + 'output/' + self.survey + '/files/' + output_name + '_d2d' + '.csv', index=False)

        # Import temp files
        path_unmatched_short = self.temp_path + self.survey + '/unmatched/'
        all_files = glob.glob(os.path.join(path_unmatched_short, "*.csv"))
        df_unmatched_short = pd.concat((pd.read_csv(f) for f in all_files), axis=0)
        # Remove duplicates
        scatalog = SkyCoord(ra=df_unmatched_short['RA'].values, dec=df_unmatched_short['DEC'].values, unit='deg')
        isdup = self.check_dup_coord(coords=scatalog)
        df_unmatched_short['isdup'] = isdup
        df_unmatched_short.query("isdup == False", inplace=True)
        df_unmatched_short.drop(columns=['isdup'], inplace=True)
        # Save file
        df_unmatched_short.to_csv(self._path + 'output/' + self.survey + '/files/' + output_name + '_unmatched' + '.csv', index=False)





class LearningAlgorithms(object):

    def __init__(self, survey, bands, path_to_csv, output_name, validation_set = False):
        """[summary]

        Args:
            survey ([type]): [description]
            bands ([type]): [description]
            temp_path ([type]): [description]
        """

        self._path = os.getcwd() + '/'
        self.survey = survey
        self._bands = bands
        self.df = pd.read_csv(path_to_csv)
        self.output_path = self._path + 'output/' + self.survey + '/'       
        self.output_name = output_name
        # self.y = self.df[['Z_SPEC']].copy().to_numpy().flatten()
        # self.X = self.df
        # self.X.drop(columns=['Z_SPEC'], inplace=True)
        
        self.train, self.test = train_test_split(self.df, test_size = 0.2, random_state=0)
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state=0)
        self.X_train = self.train.iloc[:,:-1]
        self.y_train = self.train.iloc[:,-1]
        self.X_test = self.test.iloc[:,:-1]
        self.y_test = self.test.iloc[:,-1]
        self.val = validation_set
        if validation_set == True:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=0)
        # self.output_name = output_name
        # self.mag_type = mag_type
        # self.dataset = pd.read_csv(self._path + 'output/' + self.survey + '/files/' + self.output_name + '.csv')
        # self.X = self.dataset[['%s_%s'%(mag, band) for mag in self.mag_type for band in self._bands]].copy()
        # self.y = self.dataset[['Z_SPEC']].copy()

    def data(self):
        if self.val == True:
            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        else:
            return self.X_train, self.X_test, self.y_train, self.y_test


    def sigma_eta(self, y_test, y_pred):
        delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
        sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
        outlier = np.abs(delta_z) > 0.15
        outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
        return sigma, outlier_rate


    def missing_data(self, dataset):
        all_data_na = (dataset.isnull().sum() / len(dataset)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
        print(missing_data.head(15))


    def permutation_importance(self, model, method):
        perm = PermutationImportance(model, random_state=0).fit(self.X_test, self.y_test)
        eli5.show_weights(perm, feature_names = self.X_test.columns.tolist())
        # plt.savefig(self.output_path + 'figures/' + self.output_name + '_' + method + '_feature_importance' + '.pdf', bbox_inches='tight', transparent=True)



    def plot_corrmat(self):
        corrmat = self.df.corr()
        plt.subplots(figsize=(12,9))
        sns.heatmap(corrmat, vmax=1,cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12})
        plt.savefig(self.output_path + 'figures/'  + self.output_name + '_corrmat' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()


    def plot_zphot_zspec(self, y_pred, method, lim):
        fig = plt.figure(figsize=(7,6), tight_layout=False)
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.set_xlabel(r'spectral redshift', fontsize=20)
        ax.set_ylabel(r'photometric redshift prediction', fontsize=20)

        ax.set_title('Photometric vs spectral redshift (%s)'%method, fontsize=20)
        im = ax.hist2d([float(y) for y in self.y_test], [float(y) for y in y_pred], bins=(50, 50), cmap='Blues')
        ax.set_xlim([0, lim])
        ax.set_ylim([0, lim])
        cbar = fig.colorbar(im[3])

        sigma, eta = self.sigma_eta(self.y_test, y_pred)

        ax.text(0.1,lim-0.2,r'$\sigma = {:.3f}$'.format(sigma)+'\n'+r'$\eta = {:.2f}$ %'.format(eta*100), size=15,va="baseline", ha="left", multialignment="left")

        x = np.linspace(0, 3.5, 1000)
        ax.plot(x,x, linewidth=1, color='k', linestyle='--')
        plt.savefig(self.output_path + 'figures/' + self.output_name + '_' + method + '_zphot_zspec' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()

    def sigma_eta_score(self, y_test, y_pred):
        delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
        sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
        outlier = np.abs(delta_z) > 0.15
        outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
        return sigma + outlier_rate

    def cross_validation(self, model, params={}, cv=4, scaler=False, linear=False, ann=False):
        """[summary]

        Args:
            metric ([type]): [description]
            model ([type]): [description]
            params ([type]): [description]
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: [description]
        """
        
        df = self.df.sample(frac=1).reset_index(drop=True)
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]

        df_X_list = []
        df_y_list = []
        n = len(X)
        for i in range(0,cv):
            if i < cv-1:
                df_X_list.append(X.iloc[i*int(n/cv):(i+1)*int(n/cv), :])
                df_y_list.append(y.iloc[i*int(n/cv):(i+1)*int(n/cv)])
            if i == cv-1:
                df_X_list.append(X.iloc[i*int(n/cv):, :])
                df_y_list.append(y.iloc[i*int(n/cv):])
        
        sigma_list, eta_list = [], []
        for i in tqdm(range(0,cv)):

            X_test = df_X_list[i]
            y_test = df_y_list[i]
            X_train = pd.concat(np.array([(j,X) for (j,X) in enumerate(df_X_list) if i != j])[:,1])
            y_train = pd.concat(np.array([(j,y) for (j,y) in enumerate(df_y_list) if i != j])[:,1])


            if scaler == True:
                sc_X = StandardScaler()
                X_train = sc_X.fit_transform(X_train)
                X_test = sc_X.transform(X_test)

            if ann == True:
                regressor = model
                regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)
            else:
                if linear == True:
                    regressor = make_pipeline(RobustScaler(), model(**params))
                    regressor.fit(X_train, y_train)
                elif linear == False:
                    regressor = model(**params)
                    regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            sigma, eta = self.sigma_eta(y_test, y_pred.flatten())
            sigma_list.append(sigma)
            eta_list.append(eta)
        
        return np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)

    def model(self, regressor, scaler=False):
        if scaler == False:
            regressor.fit(self.X_train, self.y_train)
            y_pred = regressor.predict(self.X_test)
        if scaler == True:
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(self.X_train)
            X_test = sc_X.transform(self.X_test)
            regressor.fit(X_train, self.y_train)
            y_pred = regressor.predict(X_test)
        y_test = self.y_test

        df = pd.DataFrame(data={'y_pred': y_pred, 'y_test': y_test})
        df.to_csv(self.output_path + 'files/ML/' + self.method + '_prediction_' + self.output_name + '.csv', index = False)

        return regressor, y_pred, y_test
        

class RandomForest(LearningAlgorithms):

    def __init__(self, survey, bands, path_to_csv, output_name):
        super().__init__(survey, bands, path_to_csv, output_name)
        self.method = 'RF'
        self.params = {'n_estimators': 86, 'max_depth': 11, 'n_jobs':-1}


    def model(self):
        """[summary]

        Returns:
            [type]: regressor, y_pred, y_test
        """
        regressor = RandomForestRegressor(**self.params)

        return LearningAlgorithms.model(self, regressor)


    def score(self, cv=4):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=RandomForestRegressor, params=self.params, cv=cv)


    def plot(self, lim):
        regressor, y_pred, _ = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, self.method, lim)
        LearningAlgorithms.permutation_importance(self, model=regressor, method=self.method)
        


class SupportVectorRegression(LearningAlgorithms):

    def __init__(self, survey, bands, path_to_csv, output_name):
        super().__init__(survey, bands, path_to_csv, output_name)
        self.method = 'SVR'
        self.params = {'C':1.0, 'cache_size':200, 'coef0':0.0, 'epsilon':0.1, 'gamma':'scale', 'kernel':'rbf', 'max_iter':-1, 'shrinking':True, 'tol':0.001}


    def model(self):
        regressor = SVR(**self.params)

        return LearningAlgorithms.model(self, regressor, scaler=True)


    def score(self, cv=4):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=SVR, params=self.params, cv=cv)


    def plot(self, lim):
        regressor, y_pred, _ = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, self.method, lim)
        LearningAlgorithms.permutation_importance(self, model=regressor, method=self.method)


class LightGBM(LearningAlgorithms):

    def __init__(self, survey, bands, path_to_csv, output_name):
        super().__init__(survey, bands, path_to_csv, output_name)
        self.method = 'LGB'
        self.params = {'objective':'regression','num_leaves':5,
                              'learning_rate':0.05, 'n_estimators':720,
                              'max_bin' : 55, 'bagging_fraction' : 0.8,
                              'bagging_freq' : 5, 'feature_fraction' : 0.2319,
                              'feature_fraction_seed':9, 'bagging_seed':9,
                              'min_data_in_leaf' :6, 'min_sum_hessian_in_leaf' : 11}


    def model(self):
        
        
        regressor = lgb.LGBMRegressor(**self.params)

        return LearningAlgorithms.model(self, regressor, scaler=False)


    def score(self, cv=4):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=lgb.LGBMRegressor, params=self.params, cv=cv)


    def plot(self, lim):
        regressor, y_pred, _ = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, self.method, lim)
        LearningAlgorithms.permutation_importance(self, model=regressor, method=self.method)

class XGBoost(LearningAlgorithms):

    def __init__(self, survey, bands, path_to_csv, output_name):
        super().__init__(survey, bands, path_to_csv, output_name)
        self.method = 'XGB'
        self.params = {'max_depth':3, 'n_estimators':2200, 'random_state':0, 'nthread': -1, 'n_jobs':-1}


    def model(self):

        regressor = xgb.XGBRegressor(**self.params)
        return LearningAlgorithms.model(self, regressor, scaler=False)


    def score(self, cv=4):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=xgb.XGBRegressor, params=self.params, cv=cv)


    def plot(self, lim):
        regressor, y_pred, _ = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, self.method, lim)
        LearningAlgorithms.permutation_importance(self, model=regressor, method=self.method)


class GradientBoostingRegression(LearningAlgorithms):

    def __init__(self, survey, bands, path_to_csv, output_name):
        super().__init__(survey, bands, path_to_csv, output_name)
        self.method = 'GBR'
        self.params = {'n_estimators':3000, 'learning_rate':0.05,
                                   'max_depth':4, 'max_features':'sqrt',
                                   'random_state':0}


    def model(self):

        regressor = GradientBoostingRegressor(**self.params)

        return LearningAlgorithms.model(self, regressor, scaler=False)


    def score(self, cv=4):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=GradientBoostingRegressor, params=self.params, cv=cv)


    def plot(self, lim):
        regressor, y_pred, _ = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, self.method, lim)
        LearningAlgorithms.permutation_importance(self, model=regressor, method=self.method)


class KernelRidgeRegression(LearningAlgorithms):

    def __init__(self, survey, bands, path_to_csv, output_name):
        super().__init__(survey, bands, path_to_csv, output_name)
        self.method = 'KRR'
        self.params = {'alpha':0.6, 'kernel':'polynomial', 'degree':2, 'coef0':2.5}

    
    def model(self):

        regressor = KernelRidge(**self.params)

        return LearningAlgorithms.model(self, regressor, scaler=True)


    def score(self, cv=4):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=KernelRidge, params=self.params, cv=cv)


    def plot(self, lim):
        regressor, y_pred, _ = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, self.method, lim)
        LearningAlgorithms.permutation_importance(self, model=regressor, method=self.method)

class ElasticNetRegression(LearningAlgorithms):

    def __init__(self, survey, bands, path_to_csv, output_name):
        super().__init__(survey, bands, path_to_csv, output_name)
        self.method = 'ENET'
        self.params={'alpha':0.0005, 'l1_ratio':.9, 'random_state':0}


    def model(self):

        regressor = make_pipeline(RobustScaler(), ElasticNet(**self.params))
        return LearningAlgorithms.model(self, regressor, scaler=False)


    def score(self, cv=4):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=ElasticNet, params=self.params, cv=cv, linear=True)


    def plot(self, lim):
        regressor, y_pred, _ = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, self.method, lim)
        LearningAlgorithms.permutation_importance(self, model=regressor, method=self.method)


class LassoRegression(LearningAlgorithms):

    def __init__(self, survey, bands, path_to_csv, output_name):
        super().__init__(survey, bands, path_to_csv, output_name)
        self.method = 'LASSO'
        self.params={'alpha': 0.0005, 'random_state':0}

    
    def model(self):

        regressor = make_pipeline(RobustScaler(), Lasso(**self.params))
        return LearningAlgorithms.model(self, regressor, scaler=False)


    def score(self, cv=4):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=Lasso, params=self.params, cv=cv, linear=True)


    def plot(self, lim):
        regressor, y_pred, _ = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, self.method, lim)
        LearningAlgorithms.permutation_importance(self, model=regressor, method=self.method)


class ArtificialNeuralNetwork(LearningAlgorithms):

    def __init__(self, survey, bands, path_to_csv, output_name):
        super().__init__(survey, bands, path_to_csv, output_name)
        self.method = 'ANN'

    def ann(self):
        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=14, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=1))
        ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

        return ann

    def model(self):

        sc = StandardScaler()
        X_train = sc.fit_transform(self.X_train)
        X_test = sc.transform(self.X_test)

        ann = self.ann()
        
        ann.fit(X_train, self.y_train, batch_size = 32, epochs = 100)
        y_pred = ann.predict(X_test)

        y_test = self.y_test
        df = pd.DataFrame(data={'y_pred': y_pred.flatten(), 'y_test': y_test})
        df.to_csv(self.output_path + 'files/ML/' + self.method + '_prediction_' + self.output_name + '.csv', index = False)

        return ann, y_pred, y_test


    def score(self, cv=4):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=self.ann(), cv=cv, scaler=True, ann=True)


    def plot(self, lim):
        regressor, y_pred, _ = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred.flatten(), self.method, lim)
        # LearningAlgorithms.permutation_importance(self, model=regressor, method=self.method)


class ConvolutionalNeuralNetwork(LearningAlgorithms):

    def __init__(self, survey, bands, path_to_csv, output_name):
        super().__init__(survey, bands, path_to_csv, output_name)
        self.X = np.load(self._path + 'output/' + self.survey + '/files/' + output_name + '.npy')
        self.X_train, self.X_test = train_test_split(self.X, test_size = 0.2, random_state=0)
        self.X_train, self.X_val = train_test_split(self.X_train, test_size = 0.2, random_state=0)
        self.method = 'CNN'


    def model(self):
        model = tfk.models.Sequential()
        model.add(tfk.layers.Conv2D(32, kernel_size=5, padding='same',
                                    input_shape=(51,51,len(self._bands)), activation='elu', strides=2))
        model.add(tfk.layers.BatchNormalization())

        model.add(tfk.layers.Conv2D(64, kernel_size=3, padding='same', 
                                activation='elu'))
        model.add(tfk.layers.BatchNormalization())

        model.add(tfk.layers.Conv2D(128, kernel_size=3, padding='same', strides=2, 
                                    activation='elu'))
        model.add(tfk.layers.BatchNormalization())  

        model.add(tfk.layers.Conv2D(256, kernel_size=3, padding='same', 
                                    activation='elu', strides=2))
        model.add(tfk.layers.BatchNormalization())

        model.add(tfk.layers.Conv2D(512, kernel_size=3, padding='same', 
                                    activation='elu', strides=2))
        model.add(tfk.layers.BatchNormalization())

        model.add(tfk.layers.Flatten())
        model.add(tfk.layers.Dense(512))
        model.add(tfk.layers.Activation('relu'))
        model.add(tfk.layers.Dense(256))
        model.add(tfk.layers.Activation('relu'))
        model.add(tfk.layers.Dense(1))

        model.compile(optimizer='adam', loss=tfk.metrics.mse)
        
        return model

    def cnn_model(self):
        scaling_train, scaling_test, scaling_val = [], [], []
        X_train = np.ndarray((len(self.X_train),51,51,len(self._bands)))
        X_val = np.ndarray((len(self.X_val),51,51,len(self._bands)))
        X_test = np.ndarray((len(self.X_test),51,51,len(self._bands)))

        for i in range(len(self._bands)):
            sigma_train = mad_std(self.X_train[...,i].flatten())
            scaling_train.append(sigma_train)
            sigma_val = mad_std(self.X_val[...,i].flatten())
            scaling_val.append(sigma_val)
            sigma_test = mad_std(self.X_test[...,i].flatten())
            scaling_test.append(sigma_test)

        for i in range(len(self.X_train)):
            for j in range(len(self._bands)):
                X_train[i,:,:,j] = np.arcsinh(self.X_train[i,:,:,j]/ scaling_train[j])

        for i in range(len(self.X_val)):
            for j in range(len(self._bands)):
                X_val[i,:,:,j] = np.arcsinh(self.X_val[i,:,:,j]/ scaling_val[j])

        for i in range(len(self.X_test)):
            for j in range(len(self._bands)):
                X_test[i,:,:,j] = np.arcsinh(self.X_test[i,:,:,j]/ scaling_test[j])

            
        # We define the batch size
        BATCH_SIZE = 64

        # Learning rate schedule
        LEARNING_RATE=0.001
        LEARNING_RATE_EXP_DECAY=0.9

        lr_decay = tfk.callbacks.LearningRateScheduler(
            lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY**epoch,
            verbose=True)
        
        # We are ready to train our model
        model = self.model()
        model.fit(X_train, self.y_train,
                steps_per_epoch=20000//BATCH_SIZE, 
                epochs=15,
                callbacks=[lr_decay], validation_data=(X_val, self.y_val))
        
        y_pred = model.predict(X_test).flatten()

        y_test = self.y_test
        df = pd.DataFrame(data={'y_pred': y_pred, 'y_test': y_test})
        df.to_csv(self.output_path + 'files/ML/' + self.method + '_prediction_' + self.output_name + '.csv', index = False)

        return model, y_pred



class Optimizer(LearningAlgorithms):

    def __init__(self, survey, bands, path_to_csv, output_name, validation_set):
        super().__init__(survey, bands, path_to_csv, output_name, validation_set=validation_set)

    def sigma_eta_score(self, y_test, y_pred):
        delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
        sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
        outlier = np.abs(delta_z) > 0.15
        outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
        return sigma + outlier_rate

    def cross_validation(self, metric, model, cv=4, scaler=False, linear=False, ann=False):
        """[summary]

        Args:
            metric ([type]): [description]
            model ([type]): [description]
            params ([type]): [description]
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: [description]
        """

        df_X_list = []
        df_y_list = []
        n = len(self.X_train)
        for i in range(0,cv):
            if i < cv-1:
                df_X_list.append(self.X_train.iloc[i*int(n/cv):(i+1)*int(n/cv), :])
                df_y_list.append(self.y_train.iloc[i*int(n/cv):(i+1)*int(n/cv)])
            if i == cv-1:
                df_X_list.append(self.X_train.iloc[i*int(n/cv):, :])
                df_y_list.append(self.y_train.iloc[i*int(n/cv):])
        
        score_list = []
        for i in range(0,cv):
            X_test = df_X_list[i]
            y_test = df_y_list[i]
            X_train = pd.concat(np.array([(j,X) for (j,X) in enumerate(df_X_list) if i != j])[:,1])
            y_train = pd.concat(np.array([(j,y) for (j,y) in enumerate(df_y_list) if i != j])[:,1])

            if scaler == True:
                sc_X = StandardScaler()
                X_train = sc_X.fit_transform(X_train)
                X_test = sc_X.transform(X_test)

            if ann == True:
                model.fit(X_train, y_train, batch_size = 32, epochs = 100)
            else:
                if linear == True:
                    model = make_pipeline(RobustScaler(), model)
                    model.fit(X_train, y_train)
                    
                elif linear == False:
                    # regressor = model
                    # regressor.set_params(**params)
                    model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            score = metric(y_test, y_pred)
            score_list.append(score)

        return np.array(score)
        

class RandomForestOptimizer(Optimizer):

    def __init__(self, survey, bands, path_to_csv, output_name, validation_set):
        super().__init__(survey, bands, path_to_csv, output_name, validation_set=validation_set)
        self.method = 'RF_Opt'


    def objective(self, params):

        est=int(params['n_estimators'])
        md=int(params['max_depth'])
        msl=int(params['min_samples_leaf'])
        mss=int(params['min_samples_split'])
        cri=params['criterion']
        params = {'n_estimators':est,'max_depth':md,'min_samples_leaf':msl,'min_samples_split':mss, 'criterion':cri, 'n_jobs':-1}

        # model=RandomForestRegressor(n_estimators=est,max_depth=md,min_samples_leaf=msl,min_samples_split=mss, criterion=cri)
        # model.fit(self.X_train,self.y_train)
        # y_pred = model.predict(self.X_val)
        # score = Optimizer.sigma_eta_score(self, self.y_val, y_pred)

        score =  Optimizer.cross_validation(self, model=RandomForestRegressor(**params), metric=self.sigma_eta_score, cv=4)
        return score.mean()

    def optimize(self, trial, max_evals = 200):

        params={'n_estimators':hp.uniform('n_estimators',10,500),
                'criterion':hp.choice('criterion', ['mse', 'mae']),
            'max_depth':hp.uniform('max_depth',5,20),
            'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
            'min_samples_split':hp.uniform('min_samples_split',2,6)}
        best=fmin(fn=self.objective,space=params,algo=tpe.suggest,trials=trial,max_evals=max_evals,rstate=np.random.RandomState(seed=2))

        criterion = ['mse', 'mae']
        best['max_depth'] = int(best['max_depth'])
        best['min_samples_leaf'] = int(best['min_samples_leaf'])
        best['min_samples_split'] = int(best['min_samples_split'])
        best['n_estimators'] = int(best['n_estimators'])
        best['criterion'] = criterion[best['criterion']]
        return best

    def best_params(self, max_evals=200):
        trial = Trials()
        best = self.optimize(trial, max_evals)
        _, y_pred, y_test = LearningAlgorithms.model(self, regressor=RandomForestRegressor(**best), scaler=True)
        eta, sigma = LearningAlgorithms.sigma_eta(self, y_test, y_pred)

        LearningAlgorithms.plot_zphot_zspec(self, y_pred.flatten(), self.method, lim=1.8)

        return y_pred, eta, sigma



class SVROptimizer(Optimizer):

    def __init__(self, survey, bands, path_to_csv, output_name, validation_set):
        super().__init__(survey, bands, path_to_csv, output_name, validation_set)
        self.method = 'SVR_Opt'

    def objective(self, params):

        gam = params['gamma']
        cc = params['C']
        params = {'kernel':'rbf', 'gamma':gam, 'C':cc}

        # model = SVR(**params)
        # model.fit(self.X_train,self.y_train)
        # y_pred = model.predict(self.X_val)
        # score = Optimizer.sigma_eta_score(self, self.y_val, y_pred)

        score =  Optimizer.cross_validation(self, model=SVR(**params), metric=self.sigma_eta_score, cv=4, scaler=True)
        return score.mean()

    def optimize(self, trial, max_evals = 200):

        params = {'C':hp.lognormal('C',0,1),
            'gamma':hp.uniform('gamma',0.001,10000)}

        best=fmin(fn=self.objective,space=params,algo=tpe.suggest,trials=trial,max_evals=max_evals,rstate=np.random.RandomState(seed=2))
        return best

    def best_params(self, max_evals=200):
        trial = Trials()
        best = self.optimize(trial, max_evals)
        _, y_pred, y_test = LearningAlgorithms.model(self, regressor=SVR(**best), scaler=True)
        eta, sigma = LearningAlgorithms.sigma_eta(self, y_test, y_pred)

        LearningAlgorithms.plot_zphot_zspec(self, y_pred.flatten(), self.method, lim=1.8)

        return y_pred, eta, sigma


class XGBoostOptimizer(Optimizer):

    def __init__(self, survey, bands, path_to_csv, output_name, validation_set):
        super().__init__(survey, bands, path_to_csv, output_name, validation_set)

    
    def objective(self, params):
        est=int(params['n_estimators'])
        md=int(params['max_depth'])
        learning=params['learning_rate']
        
        params = {'n_estimators': est, 'max_depth':md, 'learning_rate': learning}

        score =  Optimizer.cross_validation(self, model=xgb.XGBRegressor(**params), metric=self.sigma_eta_score, cv=4)
        return score.mean()

    def optimize(self, trial, max_evals=200):
        params={'n_estimators':hp.uniform('n_estimators',100,500),
            'max_depth':hp.uniform('max_depth',5,20),
            'learning_rate':hp.uniform('learning_rate',0.01,0.1)}
        best=fmin(fn=self.objective,space=params,algo=tpe.suggest,trials=trial,max_evals=max_evals,rstate=np.random.RandomState(seed=2))
        best['n_estimators'] = int(best['n_estimators'])
        best['max_depth'] = int(best['max_depth'])
        return best

    def best_params(self, max_evals=200):
        trial = Trials()
        best = self.optimize(trial, max_evals)
        _, y_pred, y_test = LearningAlgorithms.model(self, regressor=XGBoost(**best))
        eta, sigma = LearningAlgorithms.sigma_eta(self, y_test, y_pred)

        LearningAlgorithms.plot_zphot_zspec(self, y_pred.flatten(), self.method, lim=1.8)

        return y_pred, eta, sigma


class KRROptimizer(Optimizer):

    def __init__(self, survey, bands, path_to_csv, output_name, validation_set):
        super().__init__(survey, bands, path_to_csv, output_name, validation_set)
        self.method = 'KRR_Opt'

    def objective(self, params):

        gam = params['gamma']
        al= params['alpha']
        deg = params['degree']
        params = {'kernel':'polynomial', 'gamma':gam, 'alpha':al, 'degree':deg}

        # model = SVR(**params)
        # model.fit(self.X_train,self.y_train)
        # y_pred = model.predict(self.X_val)
        # score = Optimizer.sigma_eta_score(self, self.y_val, y_pred)

        score =  Optimizer.cross_validation(self, model=KernelRidge(**params), metric=self.sigma_eta_score, cv=4, scaler=True)
        return score.mean()

    def optimize(self, trial, max_evals = 200):

        params = {'alpha':hp.lognormal('alpha',0,1),
            'gamma':hp.uniform('gamma',0.001,10000),
            'degree': hp.choice('degree', [1,3,5,7,9])}

        best=fmin(fn=self.objective,space=params,algo=tpe.suggest,trials=trial,max_evals=max_evals,rstate=np.random.RandomState(seed=2))
        return best

    def best_params(self, max_evals=200):
        trial = Trials()
        best = self.optimize(trial, max_evals)
        _, y_pred, y_test = LearningAlgorithms.model(self, regressor=KernelRidge(**best), scaler=True)
        eta, sigma = LearningAlgorithms.sigma_eta(self, y_test, y_pred)

        LearningAlgorithms.plot_zphot_zspec(self, y_pred.flatten(), self.method, lim=1.8)

        return y_pred, eta, sigma



class ANNOptimizer(Optimizer):

    def __init__(self, survey, bands, path_to_csv, output_name, validation_set):
        super().__init__(survey, bands, path_to_csv, output_name, validation_set)
        self.method = 'ANN_Opt'

    # @staticmethod
    # def data():
    #     import pandas as pd
    #     import numpy as np
    #     from sklearn.model_selection import train_test_split


    #     path = os.getcwd() + '/'
    #     dataset = pd.read_csv(path + 'catalogs/' + 'MediumDeep_CFHT_CFIS_R_matched_catalog_2' + '.csv')
    #     X = dataset.iloc[:, :-1].values
    #     y = dataset.iloc[:, -1].values
    #     y = y.reshape(len(y),1)

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
    #     X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    #     sc = StandardScaler()
    #     X_train = sc.fit_transform(X_train)
    #     X_test = sc.transform(X_test)

    #     print(len(X_val), len(Y_val))

    #     return X_train, Y_train, X_val, Y_val

    # @staticmethod
    # def model(X_train, Y_train, X_val, Y_val):

    #     model = tf.keras.models.Sequential()
    #     model.add(tf.keras.layers.Dense({{choice([128, 256, 512, 1024])}}, input_shape=(len(X_train[0]),)))
    #     model.add(tf.keras.layers.Activation({{choice(['relu', 'sigmoid'])}}))
    #     model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))
    #     model.add(tf.keras.layers.Dense({{choice([128, 256, 512, 1024])}}))
    #     model.add(tf.keras.layers.Activation({{choice(['relu', 'sigmoid'])}}))
    #     model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))


    #     if {{choice(['two', 'three'])}} == 'three':
    #         model.add(tf.keras.layers.Dense({{choice([128, 256, 512, 1024])}}))
    #         model.add(tf.keras.layers.Activation({{choice(['relu', 'sigmoid'])}}))
    #         model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))
            
    #     model.add(tf.keras.layers.Dense(units=1))
    #     adam = keras.optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}})
    #     rmsprop = keras.optimizers.RMSprop(lr={{choice([10**-3, 10**-2, 10**-1])}})
    #     sgd = keras.optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}})

    #     choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}

    #     if choiceval == 'adam':
    #         optim = adam
    #     elif choiceval == 'rmsprop':
    #         optim = rmsprop
    #     else:
    #         optim = sgd
            
    #     model.compile(loss = 'mean_squared_error', metrics=['mean_squared_error'],optimizer=optim)
    #     model.fit(X_train, Y_train,
    #             batch_size={{choice([128,256,512])}},
    #             epochs=50,
    #             verbose=2,
    #             validation_data=(X_val, Y_val))

    #     y_pred = model.predict(X_val)
    #     y_test = Y_val
    #     delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
    #     sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
    #     outlier = np.abs(delta_z) > 0.15
    #     outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
    #     err = outlier_rate + sigma
    #     print('Test mean absolute error:', err)
    #     return {'loss': err, 'status': STATUS_OK, 'model': model}


    def best_params(self, max_evals = 200):


        trials = Trials()
        best_run, best_model = optim.minimize(model=model,data=data,algo=tpe.suggest, max_evals=max_evals, trials=trials)
        print(best_run)
        y_pred = best_model.predict(self.X_test, verbose = 0)
        eta, sigma = LearningAlgorithms.sigma_eta(self, self.y_test.values, y_pred)

        LearningAlgorithms.plot_zphot_zspec(self, y_pred.flatten(), self.method, lim=1.8)

        return y_pred, eta, sigma




def sigma_eta_score(y_test, y_pred):
    delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
    sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
    outlier = np.abs(delta_z) > 0.15
    outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
    return sigma + outlier_rate


def data():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split


    path = os.getcwd() + '/'
    dataset = pd.read_csv(path + 'catalogs/' + 'MediumDeep_CFHT_CFIS_R_matched_catalog_2' + '.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    y = y.reshape(len(y),1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print(len(X_val), len(Y_val))

    return X_train, Y_train, X_val, Y_val


def model(X_train, Y_train, X_val, Y_val):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=5, activation='relu'))
    model.add(tf.keras.layers.Dense(units=6, activation='relu'))
    model.add(tf.keras.layers.Dense(units=14, activation='relu'))
    model.add(tf.keras.layers.Dense(units=9, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, Y_train, batch_size = {{choice([32,64,128,256,512])}}, epochs = 100, verbose=0)
    # score, err = model.evaluate(X_val, Y_val, verbose=0)
    y_pred = model.predict(X_val)
    y_test = Y_val
    delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
    sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
    outlier = np.abs(delta_z) > 0.15
    outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
    err = outlier_rate + sigma
    #err = mean_squared_error(Y_val, Y_pred)
    # print('Test mean absolute error:', err)
    return {'loss': err, 'status': STATUS_OK, 'model': model}


def sigma_eta(y_test, y_pred):
    delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
    sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
    outlier = np.abs(delta_z) > 0.15
    outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
    return sigma, outlier_rate

#------------------------------------------------------------------#
# # # # # Create catalog # # # # #
#------------------------------------------------------------------#

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nodes", required=False, type=int, nargs="?", const=1)
    parser.add_argument("-s", "--survey", required=False, type=str, nargs="?", const='test')
    parser.add_argument("-c", "--clean", required=False, type=bool, nargs="?", const=False)
    parser.add_argument("-m", "--make", required=False, type=bool, nargs="?", const=False)
    parser.add_argument("-j", "--join", required=False, type=bool, nargs="?", const=False)
    parser.add_argument("-p", "--plot", required=False, type=bool, nargs="?", const=False)
    parser.add_argument("-l", "--learning", required=False, type=bool, nargs="?", const=False)
    parser.add_argument("-o", "--optimize", required=False, type=bool, nargs="?", const=False)
    parser.add_argument("-a", "--algorithm", required=False, type=str, nargs="?", const='SVR')

    args = parser.parse_args()

#------------------------------------------------------------------#
# # # # # PS3PI # # # # #
#------------------------------------------------------------------#

    global path
    path = os.getcwd() + '/'

    if args.survey == 'test':
        print('Modules loaded properly')

    elif args.survey == 'ps3pi_cfis':
        
        spectral_path = '/home/jimenez/spectral_catalogs/'
        spectral_names = ['alldeep.egs.uniq.2012jun13']

        out_dir = os.listdir("/n17data/jimenez/shaperun/morpho/output/")[-1]
        paste_dir = os.listdir('/n17data/jimenez/shaperun/morpho/output/%s/paste_cat_runner/output/'%(out_dir))
        input_path = '/n17data/jimenez/shaperun/morpho/output/%s/paste_cat_runner/output/'%(out_dir)

        output_name = 'CFIS_matched_deep_2_3_catalog_R'
        temp_path = '/n17data/jimenez/temp/'

        bands = ['R']

        #------------------------------------------------------------------#
        # # # # # CLEAN # # # # #
        #------------------------------------------------------------------#

        if args.clean == True:
            GenFiles = GenerateFiles(args.survey, bands, temp_path)
            GenFiles.clean_temp_directories()
            GenFiles.make_directories()

        #------------------------------------------------------------------#
        # # # # # MAKE INDIVIDUAL TILE CATALOGS # # # # #
        #------------------------------------------------------------------#

        elif args.make == True:
            cat = MakeCatalogs(args.survey, bands, temp_path)
            cat.make_survey_catalog(spectral_path, spectral_names[0])
            Parallel(n_jobs=args.nodes)(delayed(cat.make_catalog)(p, paste_dir, input_path, spectral_names[0], vignet=True) for p in tqdm(range(len(paste_dir))))

        #------------------------------------------------------------------#
        # # # # # JOIN INDIVIDUAL TILE CATALOGS # # # # #
        #------------------------------------------------------------------#

        elif args.join == True:
            cat = MakeCatalogs(args.survey, bands, temp_path)
            cat.merge_catalogs(output_name, vignet=True)

        #------------------------------------------------------------------#
        # # # # # SAVE FIGURES # # # # #
        #------------------------------------------------------------------#

        elif args.plot == True:
            GenPlot = GeneratePlots(args.survey, bands, temp_path, csv_name=output_name, spectral_names=spectral_names)
            # GenPlot.plot_d2d()
            # GenPlot.plot_matched_r_i_i_z()
            # GenPlot.plot_matched_u_r_r_i()
            GenPlot.plot_matched_z_spec_hist()
            # GenPlot.plot_unmatched_r_i_i_z()
            # GenPlot.plot_unmatched_u_r_r_i()
            GenPlot.plot_unmatched_z_spec_hist()

        #------------------------------------------------------------------#
        # # # # # MACHINE LEARNING ALGORITHMS # # # # #
        #------------------------------------------------------------------#

        elif args.learning == True:
            path_to_csv = path + 'catalogs/' + 'MediumDeep_CFHT_CFIS_R_matched_catalog_2' + '.csv'
         
            ML = LearningAlgorithms(args.survey, bands, path_to_csv, output_name)
            ML.plot_corrmat()


            algs = {'RF': RandomForest, 'ANN': ArtificialNeuralNetwork, 'LASSO': LassoRegression, 'ENET': ElasticNetRegression,
                        'XGB':XGBoost, 'KRR':KernelRidgeRegression, 'SVR': SupportVectorRegression, 'LGB': LightGBM, 'GBR': GradientBoostingRegression}

            alg = algs[args.algorithm]

            def run(alg):
                method = alg(args.survey, bands, path_to_csv, output_name)
                method.plot(lim=1.8)
                score = method.score(cv=4)
                print("%s: "%args.algorithm + "Sigma: {:.3f} +- {:.4f}, outlier rate: {:.3f} +- {:.3f} % ".format(score[0], score[1], score[2]*100, score[3]*100))

            run(alg)

        #------------------------------------------------------------------#
        # # # # # OPTIMIZE LEARNING ALGORITHMS # # # # #
        #------------------------------------------------------------------#

        elif args.optimize == True:

            path_to_csv = path + 'catalogs/' + 'MediumDeep_CFHT_CFIS_R_matched_catalog_2' + '.csv'
            max_evals = 10

            algs = {'RF': RandomForestOptimizer, 'SVR': SVROptimizer, 'XGB': XGBoostOptimizer, 'KRR': KRROptimizer, 'ANN': ANNOptimizer}
            try:
                alg = algs[args.algorithm]
            except:
                raise ValueError('Method does not have an optimization algorithm')

            if args.algorithm == 'ANN':

                # ML = LearningAlgorithms(args.survey, bands, path_to_csv, output_name, validation_set=True)
                # X_train, X_val, X_test, Y_train, Y_val, Y_test = ML.data()
                # X_train, Y_train, X_val, Y_val = data()

                # trials = Trials()
                # _, best_model = optim.minimize(model=model,data=data,algo=tpe.suggest, max_evals=max_evals, trials=trials)

                # Y_pred = best_model.predict(X_test, verbose = 0)
                # print(type(Y_pred), type(Y_test))
                # sigma, eta = sigma_eta(Y_test.to_numpy().flatten(), Y_pred.flatten())
                
                # print("%s Opt : "%args.algorithm + "Sigma: {:.3f}, outlier rate: {:.3f} % ".format(sigma, eta*100))      

                # ML.plot_zphot_zspec(Y_pred.flatten(), method='ANN_Opt', lim=1.8)



                ModelOptimizer = alg(args.survey, bands, path_to_csv, output_name, validation_set=True)
                _, sigma, eta = ModelOptimizer.best_params(max_evals=10)
                print("%s Opt : "%args.algorithm + "Sigma: {:.3f}, outlier rate: {:.3f} % ".format(sigma, eta*100))      

            else:          
                ModelOptimizer = alg(args.survey, bands, path_to_csv, output_name, validation_set=False)
                _, sigma, eta = ModelOptimizer.best_params(max_evals=max_evals)
                print("%s Opt : "%args.algorithm + "Sigma: {:.3f}, outlier rate: {:.3f} % ".format(sigma, eta*100))
        
            
#------------------------------------------------------------------#
# # # # # UNIONS # # # # #
#------------------------------------------------------------------#

    elif args.survey == 'unions':
        spectral_path = '/home/mkilbing/astro/data/CFIS/spectro_surveys/'
        spectral_names = ['data_DR14_LRG_N', 'data_DR14_LRG_S', 'galaxy_DR12v5_CMASSLOWZTOT_North', 'galaxy_DR12v5_CMASSLOWZTOT_South','sdss_main_gal']
        # spectral_names = ['sdss_main_gal']

        spectral_surveys = ['SDSS', 'SDSS', 'eBOSS', 'eBOSS', 'SDSS_2']
        # spectral_surveys = ['SDSS_2']

        output_name = 'CFIS_matched_eBOSS_SDSS_catalog_RUIZ'
        # output_name = 'CFIS_matched_SDSS_2_catalog_RUIZ'

        temp_path = '/n17data/jimenez/temp/'

        bands = ['R', 'U', 'I', 'Z']


        if args.clean == True:
            GenFiles = GenerateFiles(args.survey, bands, temp_path)
            GenFiles.clean_temp_directories()
            GenFiles.make_directories()

        elif args.make == True:
            cat = MakeCatalogs(args.survey, bands, temp_path)
            # vignet = [False, False, False, False, False]
            for i in range(len(spectral_names)):
                out_dir = os.listdir("/n17data/jimenez/shaperun_unions/output_%s/"%(spectral_surveys[i]))[-1]
                paste_dir = os.listdir('/n17data/jimenez/shaperun_unions/output_%s/%s/paste_cat_runner/output/'%(spectral_surveys[i], out_dir))
                input_path = '/n17data/jimenez/shaperun_unions/output_%s/%s/paste_cat_runner/output/'%(spectral_surveys[i], out_dir)
                cat.make_survey_catalog(spectral_path, spectral_names[i])
                Parallel(n_jobs=args.nodes)(delayed(cat.make_catalog)(p, paste_dir, input_path, spectral_names[i], vignet=False) for p in tqdm(range(len(paste_dir))))

        elif args.join == True:
            cat = MakeCatalogs(args.survey, bands, temp_path)
            cat.merge_catalogs(output_name, vignet=False)

        elif args.plot == True:
            GenPlot = GeneratePlots(args.survey, bands, temp_path, csv_name=output_name, spectral_names=spectral_names)
            # GenPlot.plot_d2d()
            GenPlot.plot_matched_r_i_i_z()
            GenPlot.plot_matched_u_r_r_i()
            GenPlot.plot_matched_z_spec_hist()
            # GenPlot.plot_unmatched_r_i_i_z()
            # GenPlot.plot_unmatched_u_r_r_i()
            GenPlot.plot_unmatched_z_spec_hist()

    # if args.survey != 'unions' or args.survey != 'ps3pi_cfis':
    #     print("Survey must either be 'unions' or 'ps3pi_cfis'")
        # raise SyntaxError("Survey must either be 'unions' or 'ps3pi_cfis'")



