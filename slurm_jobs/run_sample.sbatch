#!/bin/bash
#SBATCH --job-name=sample_job
#SBATCH --output=logs/sample_job_%j.out
#SBATCH --error=logs/sample_job_%j.err
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

module load stack/2024-06 gcc/12.2.0 python/3.11.6 eth_proxy

source ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation/socratic-env/bin/activate

cd ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation

python scripts/sample_bigmath.py --output data/samples.json

deactivate