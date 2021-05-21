#!/bin/sh
#PBS -S /bin/sh
#PBS -N matching
#PBS -j oe
#PBS -l nodes=1:ppn=8,walltime=50:00:00

module () {
  eval $(/usr/bin/modulecmd bash $*)
}

# Full path to environment
export SPENV="$HOME/.conda/envs/shapepipe"

# Load moudules and activate conda environment
module load intelpython/3
module load openmpi/4.0.5
source activate $SPENV

cd /n17data/jimenez/photoz

#python photoz.py -n 8 -c True -m True -j True -s unions -i params
#python photoz.py -n 8 -c True -u True -s unions -i params
python photoz.py -p True -s unions -i params
python photoz.py -n 4 -l True -p drop -a RF -s unions -i params

exit 0 
