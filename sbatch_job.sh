#!/bin/bash
# SCRIPT NAME: get_started.sh

# Partition type
#SBATCH --partition=high

# Number of nodes
#SBATCH --nodes=1

# Number of tasks
#SBATCH --ntasks=1

# Number of tasks per node
#SBATCH --tasks-per-node=1

# Memory per node. 5 GB (In total, 10 GB)
#SBATCH --mem=5g

# Number of GPUs per node
#SBATCH --gres=gpu:1

# Select Intel nodes (with Infiniband)
#SBATCH --constraint=intel

# Modules
source /etc/profile.d/lmod.sh
source /etc/profile.d/easybuild.sh
ml load NCCL/2.3.7-CUDA-9.0.176

# Run script
module load socker
socker run mtg/carnatic_autoencoder python train.py full_dataset filelist_full_dataset.txt config_cqt.ini 
socker run mtg/carnatic_autoencoder python convert.py full_dataset filelist_full_dataset.txt config_cqt.ini 
socker run mtg/carnatic_autoencoder python convert.py full_dataset filelist_full_dataset.txt config_cqt.ini --self-sim-matrix 
socker run mtg/carnatic_autoencoder python extract_motives.py full_dataset -r 2 -th 0.01 -csv jku_csv_files.txt