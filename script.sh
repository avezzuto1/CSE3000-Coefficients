#!/bin/sh
#
#SBATCH --job-name="cse3000_initial_runs"
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-BSc-TI

module load 2023r1
module load openmpi
module load python
module load py-numpy

srun python main.py > logger.log
