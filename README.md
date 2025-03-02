# PRMs for Socratic Teacher Feedback Data Creation

This repository contains code for generating training data for PRMs in a Socratic teaching context.

## Project Structure

The repository is organized as follows:

```
PRMs-for-Socratic-Teacher-Feedback-Data-Creation/
├── scripts/            # Python scripts for data processing
│   ├── sample_bigmath.py  # Script for sampling math problems
│   └── inference.py      # Script for running model inference
├── slurm_jobs/         # SLURM batch scripts for running on the cluster
│   ├── run_sample.sbatch  # Batch script for sampling
│   └── run_inference.sbatch # Batch script for inference
├── data/               # Directory for input and output data
│   ├── samples.json    # Sampled data
│   └── predictions.json # Model predictions (generated)
├── logs/               # Output and error logs from SLURM jobs
├── socratic-env/       # Python virtual environment
├── .env                # Environment variables (contains HuggingFace token)
└── requirements.txt    # Python dependencies
```

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