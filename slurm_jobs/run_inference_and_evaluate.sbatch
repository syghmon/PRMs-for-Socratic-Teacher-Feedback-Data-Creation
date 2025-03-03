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

# Load modules
module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

# Activate the virtual environment
source ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation/socratic-env/bin/activate

# Change to the project directory
cd ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation

# Run the inference and evaluation script
python scripts/run_inference_and_evaluate.py \
  --input data/samples.json \
  --output data/predictions.json \
  --model /cluster/scratch/sstorf/DeepSeek-R1-Distill-Qwen-7B \
  --local_model \
  --num_samples 64 \
  --temperature 0.6 \
  --top_p 0.95 \
  --hf_dataset_name "Syghmon/math-inference-results" \
  --push_to_hub \
  --evaluate \
  --response_column_name "responses" \
  --ground_truth_answer_column_name "final_answer" \
  --semaphore_limit 20 \
  --skip_inference

# Deactivate the virtual environment
deactivate 