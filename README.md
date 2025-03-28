# PRMs for Socratic Teacher Feedback Data Creation

This repository contains code for generating training data for PRMs in a Socratic teaching context.


**Notes:**

- **Root Directory:** All commands should be run from the project root directory (`PRMs-for-Socratic-Teacher-Feedback-Data-Creation/`), not from subdirectories.

- **Hugging Face Authentication:** You must create an `.env` file in the project root with your Hugging Face token:
  ```
  HUGGINGFACE_TOKEN=your_huggingface_token_here
  ```
  This is required for accessing the Big-Math-RL-Verified dataset.

## Running Jobs

To submit jobs to the cluster, use the following commands:

```bash
# Navigate to the project directory
cd ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation

# Submit the sampling job
sbatch slurm_jobs/run_sample.sbatch

# Submit the inference job (after sampling is complete)
sbatch slurm_jobs/run_inference.sbatch
```

## Monitoring Jobs

To check the status of your jobs:

```bash
squeue -u $USER
```

To view the output logs:

```bash
cat logs/sample_job_*.out
cat logs/inference_job_*.out
```

To view error logs:

```bash
cat logs/sample_job_*.err
cat logs/inference_job_*.err
```

## Environment Setup

To create the Python virtual environment:

```bash
# Create and activate the virtual environment
python -m venv socratic-env
source socratic-env/bin/activate

# Upgrade pip to latest version
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

Note: The virtual environment is already set up in the SLURM scripts.

## Motivation
Effective teaching should guide students without directly providing solutions, aligning with Socratic principles to foster independent reasoning.

## Project Overview
- Generate feedback that helps students actively engage and reason through problems.
- Balance guidance so it supports high-ability students without overly assisting lower-ability students.
- Evaluate the quality and effectiveness of generated feedback against Socratic teaching standards.

## Contacts
- Ido Hakimi (ido.hakimi@ai.ethz.ch)  
- Jakub Macina (jakub.macina@ai.ethz.ch)  
- Kumar Shridhar (shkumar@ethz.ch)  
- Shehzaad Dhuliawala (sdhuliawala@ethz.ch)



```bash
python analysis.py --samples data/samples.json --results_dir results --output_dir analysis_outputs --model_tags "SmolLM2-135M-Instruct,SmolLM2-360M-Instruct,SmolLM2-1.7B-Instruct" --bins "0-10%,10-20%,20-30%,30-40%,40-50%,50-60%,60-70%,70-80%,80-90%,90-100%"
```