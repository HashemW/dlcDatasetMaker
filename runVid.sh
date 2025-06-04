#!/bin/bash

#SBATCH -N 1                  # Use 1 node
#SBATCH -n 1                  # Run 1 task (process)
#SBATCH --gres=gpu:4          # Request 4 GPUs
#SBATCH -t 24:0:0             # Time limit: 24 hours
#SBATCH -p tron               # Partition
#SBATCH --qos=high            # QoS

#SBATCH --job-name=train      # Job name
#SBATCH --cpus-per-task=16    # CPUs per task
set -e
# Log paths
#SBATCH -o /fs/nexus-scratch/hwahed/dlcHorse/logs/%j.out
#SBATCH -e /fs/nexus-scratch/hwahed/dlcHorse/logs/%j.err
VIDEO_PATH="$1"
# Run the training script
python3 /fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVProducer.py "$VIDEO_PATH"
python3 /fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVtoVidTest.py "$VIDEO_PATH"
python3 /fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVtoData.py "$VIDEO_PATH"