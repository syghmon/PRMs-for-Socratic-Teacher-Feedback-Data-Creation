#!/bin/bash
#SBATCH --job-name=ts_inference_job
#SBATCH --output=logs/ts_inference_job_%j.out
#SBATCH --error=logs/ts_inference_job_%j.err
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:40g

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

# Activate your Python virtual environment
source ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation/socratic-env/bin/activate

cd ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation

# Create scratch directory if it doesn't exist
mkdir -p /cluster/scratch/sstorf/model_cache

echo "Starting Teacher–Student inference job with model cache in scratch space"
echo "Job started at $(date)"

# Run the modified inference script that uses scratch space
python scripts/run_inference_scratch.py \
    --input data/samples.json \
    --output results/answers.json \
    --config scripts/config.json

echo "Job completed at $(date)"

deactivate 