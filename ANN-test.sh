#!/usr/bin/bash
#SBATCH --job-name ANN-test
#SBATCH --account FTA-23-21
#SBATCH --partition qcpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 128
#SBATCH --time 00:30:00
ml purge
ml OpenMPI/4.1.4-GCC-11.3.0
python it4i-ANN-test.py
