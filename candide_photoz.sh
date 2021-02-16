#!/bin/sh
#PBS -S /bin/sh
#PBS -N matching
#PBS -j oe
#PBS -l nodes=1:ppn=42,walltime=50:00:00

module () {
  eval $(/usr/bin/modulecmd bash $*)
}

# Full path to environment
export SPENV="$HOME/.conda/envs/shapepipe"
export SPDIR="$HOME/shapepipe"

# Load moudules and activate conda environment
module load intelpython/3
module load openmpi/4.0.5
source activate $SPENV

cd match_ext

python test_match_spatial_coords_PS3PI_PSF.py -n 1 -c True -s unions
python test_match_spatial_coords_PS3PI_PSF.py -n 42 -m True -s unions
#python test_match_spatial_coords_PS3PI_PSF.py -n 1 -j True -s unions
#python test_match_spatial_coords_PS3PI_PSF.py -n 1 -p True -s unions

exit 0 
