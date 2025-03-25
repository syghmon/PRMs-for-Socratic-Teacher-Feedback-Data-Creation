#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import pathlib
import tempfile
import json
from typing import Dict, Optional

# Get the project root directory
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()

# GPU environment configurations
GPU_ENVIRONMENTS = {
    "light": {
        "gpus": 1,
        "gpumem": "40g", 
        "mem_per_cpu": "16G",
        "cpus_per_task": 4,
        "time": "4:00:00",
        "tensor_parallel_size": 1  # Single GPU setup
    },
    "heavy": {
        "gpus": 2,
        "gpumem": "40g",
        "mem_per_cpu": "16G",
        "cpus_per_task": 8,
        "time": "0:30:00",
        "tensor_parallel_size": 2  # Multi-GPU setup for large models
    }
}

def setup_env_vars():
    """Set up environment variables for model caching."""
    # Set Hugging Face cache directory to scratch space
    os.environ["HF_HOME"] = "/cluster/scratch/sstorf/model_cache"
    os.environ["HF_DATASETS_CACHE"] = "/cluster/scratch/sstorf/model_cache/datasets"
    os.environ["TORCH_HOME"] = "/cluster/scratch/sstorf/model_cache/torch"
    print(f"Using scratch directory for model cache: {os.environ['HF_HOME']}")

def load_script_config(script_path: str) -> Dict:
    """Load the configuration for a script from its associated config file."""
    script_name = os.path.basename(script_path)
    if script_name.endswith('.py'):
        script_name = script_name[:-3]  # Remove .py extension
    
    config_filename = f"{script_name}_config.json"
    config_path = os.path.join(os.path.dirname(script_path), config_filename)
    
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file {config_path} not found.")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"Loaded configuration from {config_path}: {config}")
            return config
    except json.JSONDecodeError as e:
        print(f"Error parsing configuration file {config_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error reading configuration file {config_path}: {e}")
        return {}

def create_slurm_script(script_path: str, 
                       env_config: Dict,
                       job_name: Optional[str] = None):
    """Create a temporary SLURM script file."""
    
    # Extract environment configuration
    gpus = env_config.get("gpus", 1)
    gpumem = env_config.get("gpumem", "40g")
    mem_per_cpu = env_config.get("mem_per_cpu", "16G")
    cpus_per_task = env_config.get("cpus_per_task", 4)
    time = env_config.get("time", "2:00:00")
    tensor_parallel_size = env_config.get("tensor_parallel_size", 1)
    
    # Get the script name for the job name if not provided
    if job_name is None:
        script_name = os.path.basename(script_path)
        if script_name.endswith('.py'):
            script_name = script_name[:-3]  # Remove .py extension
        job_name = script_name
    
    # Set cache directories and vLLM environment variables
    env_vars = [
        "export HF_HOME=\"/cluster/scratch/sstorf/model_cache\"",
        "export HF_DATASETS_CACHE=\"/cluster/scratch/sstorf/model_cache/datasets\"",
        "export TORCH_HOME=\"/cluster/scratch/sstorf/model_cache/torch\"",
        "export VLLM_WORKER_MULTIPROC_METHOD=spawn",  # Force spawn for vLLM workers
        "export VLLM_HOST_IP=0.0.0.0"  # Allow connections from any IP
    ]
    
    env_vars_str = "\n".join(env_vars)
    
    # Load script-specific arguments from config
    script_config = load_script_config(script_path)
    script_args = []
    
    for key, value in script_config.items():
        if isinstance(value, bool):
            if value:
                script_args.append(f"--{key}")
        elif isinstance(value, list):
            script_args.append(f"--{key} {' '.join(str(item) for item in value)}")
        else:
            script_args.append(f"--{key} {value}")
    
    # Add tensor_parallel_size from the environment config
    script_args.append(f"--tensor_parallel_size {tensor_parallel_size}")
    
    script_args_str = " ".join(script_args)
    
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={gpus}
#SBATCH --gres=gpumem:{gpumem}
#SBATCH --time={time}

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

# Activate your Python virtual environment
source ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation/socratic-env/bin/activate

cd ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation

echo "Starting job: {job_name}"
echo "Job started at $(date)"

# Print GPU information
echo "======== GPU INFORMATION ========"
nvidia-smi
echo ""

# Check GPU count
echo "======== GPU COUNT CHECK ========"
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Number of visible GPUs: $NUM_GPUS"
if [ $NUM_GPUS -lt {gpus} ]; then
    echo "WARNING: Requested {gpus} GPUs but only $NUM_GPUS are visible!"
fi
echo ""

# Set cache directories
{env_vars_str}

# Run the script with arguments from config
python {script_path} {script_args_str}

echo "Job completed at $(date)"

deactivate
"""
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a temporary file for the SLURM script
    fd, temp_path = tempfile.mkstemp(suffix='.sbatch')
    with os.fdopen(fd, 'w') as f:
        f.write(slurm_content)
    
    return temp_path

def submit_slurm_job(script_path: str, environment: str, job_name: Optional[str] = None):
    """Submit a job to SLURM with the specified environment configuration."""
    
    # Validate environment
    if environment not in GPU_ENVIRONMENTS:
        print(f"Error: Unknown environment '{environment}'. Available environments: {', '.join(GPU_ENVIRONMENTS.keys())}")
        return
    
    # Get environment configuration
    env_config = GPU_ENVIRONMENTS[environment]
    
    # Create the SLURM script
    slurm_script = create_slurm_script(script_path, env_config, job_name)
    
    # Submit the job
    cmd = ["sbatch", slurm_script]
    print(f"Submitting SLURM job: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print the result
    if result.returncode == 0:
        print(f"Job submitted successfully: {result.stdout.strip()}")
    else:
        print(f"Job submission failed: {result.stderr.strip()}")
    
    # Clean up the temporary file
    os.unlink(slurm_script)

def main():
    parser = argparse.ArgumentParser(
        description="Utility for submitting script execution jobs to SLURM with predefined GPU configurations"
    )
    
    parser.add_argument("script", type=str, 
                    help="The script to run (e.g., inference.py)")
    parser.add_argument("--environment", "-e", type=str, choices=GPU_ENVIRONMENTS.keys(), default="light",
                    help="GPU environment configuration to use")
    parser.add_argument("--job-name", "-j", type=str, default=None,
                    help="Custom name for the SLURM job (defaults to script name)")
    
    args = parser.parse_args()
    
    # Resolve the script path
    if not os.path.isabs(args.script):
        script_path = os.path.join(PROJECT_ROOT, "scripts", args.script)
        # If script doesn't have .py extension, add it
        if not script_path.endswith('.py'):
            script_path += '.py'
    else:
        script_path = args.script
    
    # Check if the script exists
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} not found")
        sys.exit(1)
    
    # Submit to SLURM
    submit_slurm_job(script_path, args.environment, args.job_name)

if __name__ == "__main__":
    main() 