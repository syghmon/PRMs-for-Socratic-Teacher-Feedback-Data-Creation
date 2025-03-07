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

echo "Starting Teacher–Student inference job"
echo "Job started at $(date)"

# Run the improved inference script using a config file for all teacher/student settings.
python scripts/inference.py \
    --input data/samples.json \
    --output results/answers.json \
    --config scripts/config.json

echo "Job completed at $(date)"

deactivate
