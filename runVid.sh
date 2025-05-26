#!/bin/bash

#SBATCH -N 1                  # Use 1 node
#SBATCH -n 1                  # Run 1 task (process)
#SBATCH --gres=gpu:4          # Request 4 GPUs
#SBATCH -t 24:0:0             # Time limit: 24 hours
#SBATCH -p tron               # Partition
#SBATCH --qos=high            # QoS

#SBATCH --job-name=train      # Job name
#SBATCH --cpus-per-task=16    # CPUs per task

# Log paths
#SBATCH -o /fs/nexus-scratch/hwahed/dlcHorse/logs/%j.out
#SBATCH -e /fs/nexus-scratch/hwahed/dlcHorse/logs/%j.err

# Run the training script
python3 /fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVtoVidTest.py