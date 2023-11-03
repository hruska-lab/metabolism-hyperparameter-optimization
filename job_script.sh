#!/usr/bin/bash
#SBATCH --job-name OptunaMorganJazzy
#SBATCH --account FTA-23-21
#SBATCH --partition qcpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 128
#SBATCH --time 24:00:00
ml purge
ml OpenMPI/4.1.4-GCC-11.3.0
python ./datacytochromy/optuna-it4i.py
