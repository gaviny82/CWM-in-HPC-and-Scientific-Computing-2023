#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=integral

# Use our reservation
#SBATCH --reservation=training


module purge
module load GCC/10.3.0

export OMP_NUM_THREADS=16

echo "N = 10000"
./integral_omp < integral10000.inp

echo "N = 100000"
./integral_omp < integral100000.inp

echo "N = 1000000"
./integral_omp < integral1000000.inp
