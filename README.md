
# ShapePipe PhotoZ 
 [![python37](https://camo.githubusercontent.com/e770ba34ca059770c9cf01c46dd567c3e0574e99d8afaf8e6179e55f432129c7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e372d79656c6c6f772e737667)](https://www.python.org/)


|Version         |Date                          |
|----------------|-------------------------------|
|alpha|12/03/2021            |


[ShapePipe](https://github.com/CosmoStat/shapepipe) extension to preprocess individual tile output catalogs and compute photometric redshifts with different machine learning methods.

Currently supports UNIONS and PS3pi_CFIS tiles.

Photometric redshift functionalities are available for all catalogs. Please refer to section **Usage** for input files format.

# Disclaimer
PhotoZ is currentlty under early development and may contain bugs or instabilities. 

# Installation 
Clone or download the PhotoZ repository:

`git clone https://github.com/xavierjimenezp/shapepipe_photoz/`

[Download](https://zenodo.org/record/4600830#.YEtHcnVKjJw) catalogs.zip and unzip 

`unzip catalogs.zip`

Create a new conda environment:

`conda env create -f environment_photoz.yml`

(Additional information relative to conda environments: [click here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)) 

Activate the environment:

`source activate photoz`
or 
`conda activate photoz`

Code is ready to run !

# Quickstart

Tutorial to run a simple example.

## Preprocessing

The following commands takes the ShapePipe output tile catalogs in candide /n17data/ directory and preprocess it into a .csv catalog. Redshift distribution plots are also generated and can be found in 'output/ps3pi_cfis/[output_name]/figures/'.

> **Note**: CFIS data is not public, please put your ShapePipe output catalog into 'example/ps3pi_cfis/tile/' if you don't have access to candide and change params_ps3pi_cfis.py config file.

`python photoz.py --survey ps3pi_cfis --clean True --input params_ps3pi_cfis`

`python photoz.py --survey ps3pi_cfis --make True --input params_ps3pi_cfis`

`python photoz.py --survey ps3pi_cfis --join True --input params_ps3pi_cfis`

`python photoz.py --survey ps3pi_cfis --generate_plots True --input params_ps3pi_cfis`

## Machine learning algorithms

The following command runs the **random forest** (RF) algorithm with default hyperparameters through the catalogs/MediumDeep_CFHT_CFIS_R_matched_catalog_2.csv  catalog using the example/params_ps3pi_cfis.py config file. It hen optimizes the RF hyperparameters using [HyperOpt](https://github.com/hyperopt/hyperopt) with a pre-defined parameter grid.

`python photoz.py --survey ps3pi_cfis --learning True --preprocess drop --algorithm RF --optimize HyperOpt --input params_ps3pi_cfis`

Output files are saved in output/ps3pi_cfis/[outpu_name]/[figures, files] directories. See **Usage** for more details.

# Machine learning algorithms

PhotoZ combines state of the art machine learning algorithms and implements the most used Python libraries: `Scikit-Learn`, `TensorFlow`, `Keras`,`XGBoost`, `LightGBM`, `HyperOpt`, etc.

### Terminology
-   **RF**: random forest.
-   **ANN**: artificial neural network.
- **XGB**: XGBoost.
- **LGB**: LightGBM
- **SVR**: support vector regression. 
- **KRR**: kernel ridge regression
-   **GBR**: gradient boosting regression.
-   **LASSO**: least absolute shrinkage and selection operator.
- **ENET**: elastic net regularization.
-   **MLM**: machine-learning method, may refer to ANN, RF or any other machine learning algorithm.
-   **spec-z**: spectroscopic redshift (the  _true_  redshift or the  _target_  of the regression).
-   **photo-z**: the derived redshift value from an MLM estimator.

# Usage

The following section presents the different parameters that need to be filled in `params.py` as well as the different arguments that can be used in order to preprocess ShapePipe's individual tile catalogs from module `paste_cat_runner` and run the machine learning algorithm on those catalogs.

## Parameters

Input parameters are found in `params.py`. The following parameters are required for any process.

 - `bands`: `(list)` name of the different bands in your catalog (e.g. ['R', 'U', 'I', 'Z'])
 - `output_path`: `(optional, str)` path to save ouput files into. Defaults to current directory.
 - `output_name`: `(str)` directory in which to save output files (files also contain ouput_name).
 - `temp_path`: `(str)` path used to create temporary files (files are usually heavy).
 
### Preprocessing
Input parameters are found in `params.py`. The following parameters are required for preprocessing calculation:

 - `spectral_path`: `(str)` path to the directory containing the spectral catalogs.
 - `spectral_names`: `(list)` list containing all spectral catalog file names. Catalogs must be .fits files. Catalog headers must contain: ra, dec, z. 
 - `path_to_tile_run`: `(str)` path to the directory containing the *spectral_surveys* directories in which shapepipe was ran
 - `spectral_surveys`: `(list)` names of the spectral surveys (e.g. ['SDSS', 'eBOSS']). Must be ordered in the same way as  `spectral_names`. 
 - `input_path`: `(optional, str or bool)` replaces the path created with path_to_tile_run and spectral_surveys (example only). Defaults to False.
 - `vignet`: `(bool)` if True, PhotoZ will output a 4d npy tensor containing all the matched object images in all bands.

> **Note 1**: when running the preprocessing function `python photoz.py --make True --survey SURVNAME`, where SURVNAME is the survey name (either 'unions' or 'ps3pi_cfis'), a .csv spectral_catalog with columns RA, DEC, Z_SPEC is created from the original fits file.
> 
> **Note 2**: please ensure that the paste_cat_runner was the last Shapepipe module to run. Otherwise, PhotoZ won't read the correct file.
> 
> **Note 3**: path to paste_cat_runner is defined as `path_to_tile_run/SURVNAME/spectral_survey/output/` where SURVNAME is either 'unions' or 'ps3pi_cifs' and spectral_survey is in spectral_surveys.

### Machine learning algorithms
Input parameters are found in `params.py`. The following parameters are required for photometric redshift calculation:

 - `max_evals`: `(optinal, int)` number of evaluations for hyper parameter optimisation using `HyperOpt`. Defaults to 200.
 - `path_to_csv`: `(str)` path to pandas DataFrame (use pd.to_csv(path_to_csv, index=False) to create input catalog. Please name the spectral redshift column "Z_SPEC" and put it at the end. Defaults to None.
 - `weights`: `(bool or str)` if True, weights will be computed using r band. If str, path for .npy file. If None, won't use weights for regression. Defaults to True.
 - `cv`: `(int)` folds for cross-validation. Defaults to 10.
 - `feature_engineering`: `(bool)` If True, will try all possible color combination while keeping the r band . Defaults to False.
 - `feature_importance`: `(bool)`If True, will save a .csv file in LaTeX format with feature importance. Defaults to False.
 - `plot`: `(bool)` If True, will plot the correlation matrix and z_phot vs z_spec estimation. Defaults to False,
 - `morph_importance`: `(bool)` If True, will save a .csv filein LaTeX format with score values with and wthout morphological parameters for each possible band combination. Defaults to False.


>**Note**: If the input DataFrame in `path_to_csv` containes NaNs they will be treated accordingly by the *preprocess* function. Please refrain from setting negative values (e.g. -1, -10, -99, etc) or any other kind of value instead of NaNs. Otherwise, the *preprocess* function will ignore them and the code will perform poorly.

## Arguments

The `photoz.py` python file takes the following arguments. The function can be executed without arguments in which case the default parameters will be used:

 - `--nodes or -n`: `(optinal, int)` number of cores to be used. Defaults to 1.
> **Note**: only `--make`, `--learning` and `--optimize` can use more than one core.
 - `--survey or -s`: `(optinal, str)` survey name. Either 'unions', 'ps3pi_cfis' or other. If other,  preprocessing functions won't be available. Defaults to 'none'.
 - `--input or -i`: `(optinal, str)` input file name for parameters file. Defaults to params.

### Preprocessing

- `--clean or -c`: `(optinal, bool)` if True, will create temporary directories and remove all the files within. Will also create output directories. Defaults to False.
- `--make or -m`: `(optinal, bool)` if True, will create individual .csv files for each tile within paste_cat_runner. Defaults to False.
- `--join or -j`: `(optinal, bool)` if True, will merge the individual .csv files created by `--make`. Files will be saved in /output/SURVNAME/files/ where SURVNAME is either 'unions' or 'ps3pi_cfis'. Defaults to False.
- `--generate_plots or -g`: `(optinal, bool)` if True, will generate plots in /output/SURVNAME/figures/ where SURVNAME is either 'unions' or 'ps3pi_cfis'. Defaults to False.

### Machine learning

 - `--algorithm or -a`: `(optinal, str)` MLM algorithm (see **Terminology** for options). Use BEST if you want to try them all and output the best one. Defaults to RF.
 - `--preprocess or -p`: `(optional, str, int or float)`: method to handle missing values. Columns with>15% missing values will be droped. If type(method)=str, methods include: 'drop', 'mode', 'mean', 'median'. if type(method)=int or float, missing values will be replaced by the selected value. Defaults to None.
    - drop: rows with missing values are droped
    - mode: missing values are replaced with column mode
    - mean, median: same as mode
   
 - `--learning or -l`: `(optinal, bool)` if True, will compute **photo-z** using the MLM algorithm specified with `--algorithm`. Default hyperparameters will be used. Defaults to False.
 - `--optimize or -o`: `(optinal, str)` either 'HyperOpt', 'RandomSearch' or 'GridSearch'. Will use a predefined hyperparameter grid to optimize the MLM algorithm. For each method the grid is hard coded in Optimizer class and can be changed manually. This functionality may be very time expensive. Recommended value for max_evals in params.py is 200. Defaults to None.

>**Note**: `--optimize` currently only supports :
> - HyperOpt: RF, SVR, KRR and XGB (ANN support is deprecated) 
> - RandomSearch: RF (SVR support is deprecated). 
> - GridSearch: RF (SVR support is deprecated).

## Output files

Output files are saved in ouput/[survey]/[output_name]/[files or figures]

## Citation

Xavier Jiménez, Marin Kilbenger, Joana Frontera-Pons. Photometric redshifts for UNIONS, 2021 in prep.
