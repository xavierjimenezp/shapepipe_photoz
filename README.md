<image src="Cosmostat_logo.png" width="250" align="left"/>

# Shapepipe PhotoZ 

[![travis](https://camo.githubusercontent.com/2d96c42159599cd00db0db2d45b6f4884da049af9cc16fee8ba68d531d6bb28e/68747470733a2f2f7472617669732d63692e636f6d2f436f736d6f537461742f7368617065706970652e7376673f6272616e63683d6d6173746572)](https://travis-ci.com/CosmoStat/shapepipe) [![python35](https://camo.githubusercontent.com/8c2717c22bf2c14eea589cc3f199ceb70e7176f4b1afc2a177a3f36c716910a4/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e352d79656c6c6f772e737667)](https://www.python.org/)  [![python36](https://camo.githubusercontent.com/6b522695ff1ef1af03d049f66348ce5c1f09f6bab606522ccb1a8d27142f7b95/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e362d79656c6c6f772e737667)](https://www.python.org/) [![python37](https://camo.githubusercontent.com/e770ba34ca059770c9cf01c46dd567c3e0574e99d8afaf8e6179e55f432129c7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e372d79656c6c6f772e737667)](https://www.python.org/)


|Version         |Date                          |
|----------------|-------------------------------|
|0.0.1|16/02/2021            |


[Shapepipe](https://github.com/CosmoStat/shapepipe) extension to preprocess shapepipe individual tile output catalogs and compute photometric redshifts with different methods.

Currently supports UNIONS and CFIS_PS3PI tiles.

Photometric redshift functionalities are available for all catalogs. Please refer to section **Usage** for input files format.

# Disclaimer
PhotoZ is currentlty under early developement and may contain bugs or instabilities. 

# Installation 
Clone or download the PhotoZ repository:
`git clone https://github.com/xavierjimenezp/shapepipe_photoz/`

If you use conda, create a new environment:
`conda env create -f environment_photoz.yml`
(Additional information relative to conda environments https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)

Activate the environment:
`source activate photoz`
or 
`conda activate photoz`

Install the code:  
`python setup.py --install`

# Examples
WIP
# Algorithms
WIP
## Machine learning

### Terminology
-   **RF**: random forest.
-   **ANN**: artificial neural network.
- **XGB**: XGBoost.
- **SVR**: support vector regression. 
- **KRR**: kernel ridge regression
-   **GBR**: gradient boosting regression.
-   **LASSO**: least absolute shrinkage and selection operator.
- **ENET**: elastic net regularization.
-   **MLM**: machine-learning method, may refer to ANN,RF or any other machine learning algorithm.
-   **spec-z**: spectroscopic redshift (the  _true_  redshift or the  _target_  of the regression).
-   **photo-z**: the derived redshift value from an MLM estimator.

# Usage
## Parameters
### Preprocessing
Input parameters are found in `params.py`. The following parameters are required for preprocessing calculation:

 - `spectral_path`: `(str)` path to the directory containing the spectral catalogs.
 - `spectral_names`: `(list)` list containing all spectral catalog file names. Catalogs must be .fits files. Catalog headers must contain: ra, dec, z. 
 - `temp_path`: `(str)` path used to create temporary files (files are usually heavy).
 - `bands`: `(list)` name of the different bands in your catalog (e.g. ['R', 'U', 'I', 'Z'])
 - `path_to_tile_run`: `(str)` path to the directory containing the *spectral_surveys* directories in which shapepipe was ran
 - `spectral_surveys`: `(list)` names of the spectral surveys (e.g. ['SDSS', 'eBOSS']). Must be ordered in the same way as  `spectral_names`. 

> **Note 1**: when running the preprocessing function `python photoz.py --nodes NODENUM --make True --survey SURVNAME`, where NODEDUM is the number of cores and SURVNAME is the survey name (either 'unions' or 'ps3pi_cfis'), a .csv spectral_catalog with columns RA, DEC, Z_SPEC is created from the original fits file.
> 
> **Note 2**: please ensure that the paste_cat_runner was the last Shapepipe module to run. Otherwise, PhotoZ won't read the correct file.
> 
> **Note 3**: path to paste_cat_runner is defined as `path_to_tile_run/spectral_survey/output/` where spectral_survey is in spectral_surveys

### Machine learning algorithms
Input parameters are found in `params.py`. The following parameters are required for photometric redshift calculation:

 - `temp_path`: `(str)` path used to create temporary files (files are usually heavy).
 - `bands`: `(list)` name of the different bands in your catalog (e.g. ['R', 'U', 'I', 'Z'])
 - `max_evals`: `(optinal, int)` number of evaluations for hyper parameter optimisation using `HyperOpt`. Defaults to 200.
 - `path_to_csv`: `(str)` path to pandas DataFrame (use pd.to_csv(path_to_csv.csv, index=False) to create input catalog. Please give name the spectral redshift column "Z_SPEC" and put it at the end. See /catalogs/MediumDeep_CFHT_CFIS_R_matched_catalog_2.csv for an example.

>**Note**: If the input DataFrame in `path_to_csv` containes NaNs they will be treated accordingly by the *preprocess* function. Please refrain from setting negative values (e.g. -1, -10, -99, etc) or any other kind of value instead of NaNs. Otherwise, the *preprocess* function will ignore them and the code will perform poorly.

## Quickstart
### Preprocessing
WIP

### Machine learning




## Citation
WIP
