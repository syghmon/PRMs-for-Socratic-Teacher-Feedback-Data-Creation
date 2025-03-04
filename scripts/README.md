# Math Problems Inference and Evaluation

This directory contains scripts for generating and evaluating model responses to mathematical problems.

## Overview

The workflow consists of three main steps:

1. **Inference**: Running a model to generate responses for mathematical problems.
2. **Dataset Creation**: Converting the generated responses to a HuggingFace dataset.
3. **Evaluation**: Evaluating the correctness of responses using mathematical equivalence checking.

## Scripts

### 1. `inference.py`

Runs inference on a model to generate responses for mathematical problems. This script can be configured to generate multiple samples per problem.

```bash
python inference.py \
  --input data/samples.json \
  --output data/predictions.json \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --num_samples 64 \
  --temperature 0.6 \
  --top_p 0.95 \
  --format_for_hf
```

Key parameters:
- `--input`: Path to the JSON file of problem samples.
- `--output`: Path to save the model predictions.
- `--model`: Hugging Face model name or local path.
- `--num_samples`: Number of samples to generate per problem (default: 1).
- `--format_for_hf`: Format output for HuggingFace datasets.

### 2. `evaluate_responses.py`

Evaluates model responses against ground truth answers.

```bash
python evaluate_responses.py \
  --predictions_dataset_name username/math-inference-results \
  --response_column_name responses \
  --ground_truth_answer_column_name final_answer \
  --use_llm_judge_backup
```

Key parameters:
- `--predictions_dataset_name`: Name of the HuggingFace dataset containing predictions.
- `--response_column_name`: Column name with model responses.
- `--ground_truth_answer_column_name`: Column name with ground truth answers.
- `--use_llm_judge_backup`: Whether to use LLM judge as backup for evaluation.

### 3. `run_inference_and_evaluate.py`

Combines the inference and evaluation steps into a single workflow.

```bash
python run_inference_and_evaluate.py \
  --input data/samples.json \
  --output data/predictions.json \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --num_samples 64 \
  --hf_dataset_name username/math-inference-results \
  --push_to_hub \
  --evaluate \
  --response_column_name responses \
  --ground_truth_answer_column_name final_answer
```

Key parameters:
- Includes all parameters from both inference.py and evaluate_responses.py.
- `--push_to_hub`: Whether to push the predictions to HuggingFace Hub.
- `--evaluate`: Whether to run evaluation after inference.

## Running on Slurm

For running on a Slurm cluster, we provide Slurm job scripts in the `slurm_jobs` directory:

### 1. `run_inference.sbatch`

Runs just the inference part on Slurm:

```bash
sbatch slurm_jobs/run_inference.sbatch
```

### 2. `run_inference_and_evaluate.sbatch`

Runs the complete workflow (inference + evaluation) on Slurm with predefined parameters:

```bash
sbatch slurm_jobs/run_inference_and_evaluate.sbatch
```

## Evaluation

The evaluation process extracts final answers from model responses by looking for expressions in `\boxed{}` notation. It then compares these answers with ground truth answers using symbolic math equivalence checking.

When using `--use_llm_judge_backup`, if the direct symbolic comparison fails, an LLM judge is used as a backup to determine if the answers are equivalent. (TODO?)

## Results

After evaluation, the dataset will include additional columns:
- `responses_extracted_answers`: Extracted answers from model responses
- `responses_correctness`: Boolean values indicating correctness of each response 
