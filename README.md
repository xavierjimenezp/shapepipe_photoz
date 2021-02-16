# Shapepipe PhotoZ
Shapepipe extension to preprocess shapepipe individual tile output catalogs and compute photometric redshifts with different methods.

Currently supports UNIONS and CFIS_PS3PI tiles.

Photometric redshift functionalities are available for all catalogs. Please refer to section **Usage** for input files format.

# Disclaimer
PhotoZ is cruentlty under developement and may contain bugs or instabilities. 

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
`python setup.py install`

# Examples
WIP
# Algorithms

# Usage
### Preprocessing
WIP
### Machine learning algorithms
Input parameters are found in `params.py`. The following parameters are required for photometric redshift calculation:

 - `spectral_path (type -> str)`: [description]. 
 - `spectral_names (type -> list)`: [description]. 
 - `spectral_surveys (type -> list)`: [description]. 
 - `temp_path (type -> str)`: path used to create temporary files.
 - `bands (type -> list)`: name of the different bands in your catalog (e.g. ['R', 'U', 'I', 'Z'])
 - `max_evals (optinal, type -> int)`: [description]. Defaults to 4.
 - `path_to_csv (type -> str)`: path to pandas DataFrame (use pd.to_csv(path_to_csv.csv, index=False) to create input catalog. Please give name the spectral redshift column "Z_SPEC" and put it at the end. See /catalogs/MediumDeep_CFHT_CFIS_R_matched_catalog_2.csv for an example.

**Note**: If the input DataFrame containes NaNs they will be treated accordingly by the *preprocess* function. Please refrain from setting negative values (e.g. -1, -10, -99, etc) or any other kind of value instead of NaNs. Otherwise, the *preprocess* function will ignore them and the code will perform poorly.

The various example scripts includes comments about the different variables which the user needs to set. Each operational mode has a  _quickstart_  dedicated script as well as an  _advanced_  script. The latter include more job-options as well as more detailed documentation.

For each of the following, please follow the four respective steps (generation, training, optimization/verification, evaluation) in sequence. For instance, for single regression, do:

python scripts/annz_singleReg_quick.py --singleRegression --genInputTrees
python scripts/annz_singleReg_quick.py --singleRegression --train
python scripts/annz_singleReg_quick.py --singleRegression --optimize
python scripts/annz_singleReg_quick.py --singleRegression --evaluate






## Citation
WIP
