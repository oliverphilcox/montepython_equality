#!/bin/bash
#
#SBATCH --job-name=lensing_marg_eq
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=8
#SBATCH --time=05:59:59
#SBATCH --mem-per-cpu=500
#SBATCH --array=1-5
#SBATCH -N 1
#SBATCH --output=/home/ophilcox/out/lensing_marg_eq_%a.log

# Load modules
module load anaconda3
source activate ptenv
module load openmpi intel

MP=/home/ophilcox/PkEFT/montepython_public/montepython/MontePython.py
PARAM=/home/ophilcox/PkEFT/montepython_public/input/boss_eq_h/lensing_marg.param
OUT=/tigress/ophilcox/eq_BOSS/lensing_marg_v2
COV=/tigress/ophilcox/eq_BOSS/lensing_marg/lensing_marg.covmat
BESTFIT=/tigress/ophilcox/eq_BOSS/lensing_marg/lensing_marg.bestfit
N=50000

# buffer to avoid overwriting
let sleeper=$SLURM_ARRAY_TASK_ID*10
sleep $sleeper

# run to initialize
python $MP run -p $PARAM -o $OUT -N 5 -f 1. -c $COV -j fast -b $BESTFIT


# run properly
mpirun -np 8 python $MP run -p $PARAM -o $OUT -N $N -f 1. -c $COV -j fast -b $BESTFIT

# analyze and compute covariance
python $MP info $OUT --want-covmat
