#!/bin/bash
#SBATCH --job-name=inference_job
#SBATCH --output=logs/inference_job_%j.out
#SBATCH --error=logs/inference_job_%j.err
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

# Load modules
module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

# Activate virtual environment
source ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation/socratic-env/bin/activate

cd ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation

# Run inference using the locally transferred model
python scripts/inference.py \
  --input data/samples.json \
  --output data/predictions.json \
  --model /cluster/scratch/sstorf/DeepSeek-R1-Distill-Qwen-7B \
  --local_model

deactivate
