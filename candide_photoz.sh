#!/bin/sh
#PBS -S /bin/sh
#PBS -N matching
#PBS -j oe
#PBS -l nodes=1:ppn=46,walltime=50:00:00

module () {
  eval $(/usr/bin/modulecmd bash $*)
}

# Full path to environment
export SPENV="$HOME/.conda/envs/shapepipe"

# Load moudules and activate conda environment
module load intelpython/3
module load openmpi/4.0.5
source activate $SPENV

cd ~/photoz

python photoz.py -c True -s ps3pi_cfis -i params
python photoz.py -n 46 -m True -s ps3pi_cfis -i params

exit 0 
