#!/bin/bash
#SBATCH --job-name=math_eval
#SBATCH --output=logs/math_eval_%j.out
#SBATCH --error=logs/math_eval_%j.err
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:40g
#SBATCH --time=24:00:00

# This script takes all arguments and passes them to run_math_eval.sh
# Example usage:
# sbatch slurm_jobs/run_math_eval.sbatch -d username/math-dataset -m /path/to/model -n 64

# Load modules
module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

# Activate the virtual environment
source ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation/socratic-env/bin/activate

# Change to the project directory
cd ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation

# Make sure the logs directory exists
mkdir -p logs

# Run the evaluation script with all arguments passed to this slurm script
./scripts/run_math_eval.sh "$@"

# Deactivate the virtual environment
deactivate 