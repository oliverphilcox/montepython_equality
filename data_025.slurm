#!/bin/bash
#
#SBATCH --job-name=data_025_eq
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=8
#SBATCH --time=05:59:59
#SBATCH --mem-per-cpu=500
#SBATCH --array=1-5
#SBATCH -N 1
#SBATCH --output=/home/ophilcox/out/data_025_eq_%a.log

# Load modules
module load anaconda3
source activate ptenv
module load openmpi intel

MP=/home/ophilcox/PkEFT/montepython_public/montepython/MontePython.py
PARAM=/home/ophilcox/PkEFT/montepython_public/input/boss_eq_h/data_025.param
OUT=/tigress/ophilcox/eq_BOSS/data_025
COV=/tigress/ophilcox/eq_BOSS/v9/data_v3/data_v3.covmat
BESTFIT=/tigress/ophilcox/eq_BOSS/v9/data_v3/data_v3.bestfit
N=50000

# buffer to avoid overwriting
let sleeper=$SLURM_ARRAY_TASK_ID*10
sleep $sleeper

# run to initialize
python $MP run -p $PARAM -o $OUT -N 5 -f 1. -j fast -c $COV -b $BESTFIT

# run properly
mpirun -np 8 python $MP run -p $PARAM -o $OUT -N $N -f 1. -j fast -c $COV -b $BESTFIT

# analyze and compute covariance
python $MP info $OUT --want-covmat
