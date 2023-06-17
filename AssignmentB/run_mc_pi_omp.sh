#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=heat

# Use our reservation
#SBATCH --reservation=training



module purge
module load GCC/10.3.0

export OMP_NUM_THREADS=16
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

./mc_pi_omp
