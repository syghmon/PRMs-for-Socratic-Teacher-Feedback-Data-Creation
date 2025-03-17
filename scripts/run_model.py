#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import pathlib
import tempfile
from typing import List, Optional

# Get the project root directory
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()

def setup_env_vars(use_scratch: bool = True):
    """Set up environment variables for model caching."""
    if use_scratch:
        # Set Hugging Face cache directory to scratch space
        os.environ["TRANSFORMERS_CACHE"] = "/cluster/scratch/sstorf/model_cache"
        os.environ["HF_HOME"] = "/cluster/scratch/sstorf/model_cache"
        os.environ["HF_DATASETS_CACHE"] = "/cluster/scratch/sstorf/model_cache/datasets"
        os.environ["TORCH_HOME"] = "/cluster/scratch/sstorf/model_cache/torch"
        print(f"Using scratch directory for model cache: {os.environ['TRANSFORMERS_CACHE']}")

def run_local(script_path: str, script_args: List[str], use_scratch: bool = True):
    """Run a script locally with the correct environment."""
    setup_env_vars(use_scratch)
    
    # Construct the command
    cmd = [sys.executable, script_path] + script_args
    print(f"Running locally: {' '.join(cmd)}")
    
    # Execute the command
    subprocess.run(cmd)

def create_slurm_script(script_path: str, script_args: List[str], 
                       job_name: str = "model_job",
                       mem_per_cpu: str = "16G",
                       gpus: int = 1,
                       gpumem: str = "40g",
                       ntasks: int = 4,
                       time: str = "2:00:00"):
    """Create a temporary SLURM script file."""
    
    # Set cache directories
    env_vars = [
        "export TRANSFORMERS_CACHE=\"/cluster/scratch/sstorf/model_cache\"",
        "export HF_HOME=\"/cluster/scratch/sstorf/model_cache\"",
        "export HF_DATASETS_CACHE=\"/cluster/scratch/sstorf/model_cache/datasets\"",
        "export TORCH_HOME=\"/cluster/scratch/sstorf/model_cache/torch\""
    ]
    
    env_vars_str = "\n".join(env_vars)
    
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --ntasks={ntasks}
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

# Run the script
python {script_path} {' '.join(script_args)}

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

def submit_slurm_job(script_path: str, script_args: List[str], 
                    job_name: str = "model_job",
                    mem_per_cpu: str = "16G", 
                    gpus: int = 1,
                    gpumem: str = "40g",
                    ntasks: int = 4,
                    time: str = "2:00:00"):
    """Submit a job to SLURM."""
    # Create the SLURM script
    slurm_script = create_slurm_script(
        script_path, script_args, job_name, mem_per_cpu, gpus, gpumem, ntasks, time
    )
    
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

def run_prompt_comparison_pipeline(
    samples_path: str,
    config_path: Optional[str] = None,
    output_dir: str = "results",
    prompt_analysis_dir: str = "prompt_analysis",
    use_judge: bool = False,
    use_slurm: bool = False,
    gpus: int = 1,
    time: str = "8:00:00",
):
    """
    Run the complete prompt comparison pipeline:
    1. Run inference.py with multiple teacher prompts
    2. Run prompt_analysis.py to analyze the results
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "scripts", "config.json")
    
    # Ensure paths are absolute
    if not os.path.isabs(samples_path):
        samples_path = os.path.join(PROJECT_ROOT, samples_path)
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    if not os.path.isabs(prompt_analysis_dir):
        prompt_analysis_dir = os.path.join(PROJECT_ROOT, prompt_analysis_dir)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(prompt_analysis_dir, exist_ok=True)
    
    # Load config to extract prompt names and student models
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract student models and prompt names
    student_models = [safe_model_id(model["model_name"]) for model in config["student_models"]]
    prompt_names = [prompt["name"] for prompt in config["teacher_prompts"]]
    
    # Generate command arguments
    inference_args = [
        os.path.join(PROJECT_ROOT, "scripts", "inference.py"),
        "--input", samples_path,
        "--output", os.path.join(output_dir, "answers.json"),
        "--config", config_path
    ]
    if use_judge:
        inference_args.append("--use_judge")
    
    analysis_args = [
        os.path.join(PROJECT_ROOT, "scripts", "prompt_analysis.py"),
        "--samples", samples_path,
        "--results_dir", output_dir,
        "--output_dir", prompt_analysis_dir,
        "--student_models", ",".join(student_models),
        "--prompt_names", ",".join(prompt_names)
    ]
    
    # Run inference
    print("=== Running Inference with Multiple Teacher Prompts ===")
    if use_slurm:
        submit_slurm_job(
            script_path="scripts/inference.py",
            script_args=inference_args[1:],  # Skip the script path
            job_name="teacher_inference",
            gpus=gpus,
            time=time
        )
        print("\nInference job submitted to SLURM. Once it completes, run the analysis step with:")
        print(f"python scripts/run_model.py --run-analysis-only --samples {samples_path} --config {config_path}")
    else:
        # Run locally
        run_local(inference_args[0], inference_args[1:])
        
        # Run analysis
        print("\n=== Running Prompt Analysis ===")
        run_local(analysis_args[0], analysis_args[1:])
        
        print("\n=== Pipeline Complete ===")
        print(f"Inference results saved to: {output_dir}")
        print(f"Analysis results saved to: {prompt_analysis_dir}")

def run_analysis_only(
    samples_path: str,
    config_path: Optional[str] = None,
    output_dir: str = "results",
    prompt_analysis_dir: str = "prompt_analysis",
):
    """
    Run only the analysis part of the pipeline.
    Useful when inference was run separately (e.g., via SLURM).
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "scripts", "config.json")
    
    # Ensure paths are absolute
    if not os.path.isabs(samples_path):
        samples_path = os.path.join(PROJECT_ROOT, samples_path)
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    if not os.path.isabs(prompt_analysis_dir):
        prompt_analysis_dir = os.path.join(PROJECT_ROOT, prompt_analysis_dir)
    
    # Load config to extract prompt names and student models
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract student models and prompt names
    student_models = [safe_model_id(model["model_name"]) for model in config["student_models"]]
    prompt_names = [prompt["name"] for prompt in config["teacher_prompts"]]
    
    # Generate command arguments
    analysis_args = [
        os.path.join(PROJECT_ROOT, "scripts", "prompt_analysis.py"),
        "--samples", samples_path,
        "--results_dir", output_dir,
        "--output_dir", prompt_analysis_dir,
        "--student_models", ",".join(student_models),
        "--prompt_names", ",".join(prompt_names)
    ]
    
    # Run analysis
    print("\n=== Running Prompt Analysis ===")
    run_local(analysis_args[0], analysis_args[1:])
    
    print("\n=== Analysis Complete ===")
    print(f"Analysis results saved to: {prompt_analysis_dir}")

def safe_model_id(model_name: str) -> str:
    """
    Convert a model name (e.g. 'huggingfaceTB/SmolLM2-1.7B-Instruct')
    into a safe tag (e.g. 'SmolLM2-1.7B-Instruct') for use in filenames.
    """
    base = model_name.split("/")[-1]
    # Replace characters that might cause filename issues
    base = base.replace(" ", "_").replace(":", "_").replace("\\", "_").replace("/", "_")
    return base

def main():
    parser = argparse.ArgumentParser(
        description="Unified utility for running model scripts locally or on SLURM"
    )
    
    # Define subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Run a single script
    script_parser = subparsers.add_parser('script', help='Run a single script')
    script_parser.add_argument("script", type=str, 
                        help="The script to run (e.g., inference.py, test_model.py)")
    script_parser.add_argument("--slurm", action="store_true", 
                        help="Submit as a SLURM job instead of running locally")
    script_parser.add_argument("--job-name", type=str, default="model_job",
                        help="Name for the SLURM job")
    script_parser.add_argument("--mem-per-cpu", type=str, default="16G",
                        help="Memory per CPU core")
    script_parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs")
    script_parser.add_argument("--gpumem", type=str, default="40g",
                        help="GPU memory requirement")
    script_parser.add_argument("--ntasks", type=int, default=4,
                        help="Number of CPU tasks")
    script_parser.add_argument("--time", type=str, default="2:00:00",
                        help="Maximum job runtime (HH:MM:SS)")
    script_parser.add_argument("--no-scratch", action="store_true",
                        help="Don't use scratch directory for model cache (local mode only)")
    
    # Run the prompt comparison pipeline
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the complete prompt comparison pipeline')
    pipeline_parser.add_argument("--samples", "-s", type=str, required=True,
                        help="Path to the samples JSON file")
    pipeline_parser.add_argument("--config", "-c", type=str, 
                        help="Path to the config JSON file (default: scripts/config.json)")
    pipeline_parser.add_argument("--output-dir", "-o", type=str, default="results",
                        help="Directory to save inference results")
    pipeline_parser.add_argument("--analysis-dir", "-a", type=str, default="prompt_analysis",
                        help="Directory to save analysis results")
    pipeline_parser.add_argument("--use-judge", action="store_true",
                        help="Use LLM judge for evaluating answers")
    pipeline_parser.add_argument("--slurm", action="store_true",
                        help="Submit inference as a SLURM job")
    pipeline_parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs (for SLURM mode)")
    pipeline_parser.add_argument("--time", type=str, default="8:00:00",
                        help="Maximum job runtime (for SLURM mode)")
    
    # Run just the analysis part
    analysis_parser = subparsers.add_parser('analysis', help='Run only the prompt analysis')
    analysis_parser.add_argument("--samples", "-s", type=str, required=True,
                        help="Path to the samples JSON file")
    analysis_parser.add_argument("--config", "-c", type=str, 
                        help="Path to the config JSON file (default: scripts/config.json)")
    analysis_parser.add_argument("--output-dir", "-o", type=str, default="results",
                        help="Directory with inference results")
    analysis_parser.add_argument("--analysis-dir", "-a", type=str, default="prompt_analysis",
                        help="Directory to save analysis results")
    
    # For backward compatibility, also support the original style arguments
    parser.add_argument("--run-pipeline", action="store_true",
                        help="Run the complete prompt comparison pipeline")
    parser.add_argument("--run-analysis-only", action="store_true",
                        help="Run only the prompt analysis part")
    
    # These arguments are used with --run-pipeline and --run-analysis-only
    parser.add_argument("--samples", type=str,
                        help="Path to the samples JSON file (for pipeline mode)")
    parser.add_argument("--config", type=str,
                        help="Path to the config JSON file (for pipeline mode)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results (for pipeline mode)")
    parser.add_argument("--analysis-dir", type=str, default="prompt_analysis",
                        help="Directory to save analysis results (for pipeline mode)")
    parser.add_argument("--use-judge", action="store_true",
                        help="Use LLM judge for evaluating answers (for pipeline mode)")
    
    # All remaining arguments will be passed to the script in script mode
    args, remaining_args = parser.parse_known_args()
    
    # Handle the different modes
    if args.run_pipeline:
        # Old-style direct argument
        if args.samples is None:
            parser.error("--samples is required with --run-pipeline")
        run_prompt_comparison_pipeline(
            samples_path=args.samples,
            config_path=args.config,
            output_dir=args.output_dir,
            prompt_analysis_dir=args.analysis_dir,
            use_judge=args.use_judge,
            use_slurm=False
        )
    elif args.run_analysis_only:
        # Old-style direct argument
        if args.samples is None:
            parser.error("--samples is required with --run-analysis-only")
        run_analysis_only(
            samples_path=args.samples,
            config_path=args.config,
            output_dir=args.output_dir,
            prompt_analysis_dir=args.analysis_dir
        )
    elif args.mode == 'pipeline':
        # New-style subcommand
        run_prompt_comparison_pipeline(
            samples_path=args.samples,
            config_path=args.config,
            output_dir=args.output_dir,
            prompt_analysis_dir=args.analysis_dir,
            use_judge=args.use_judge,
            use_slurm=args.slurm,
            gpus=args.gpus,
            time=args.time
        )
    elif args.mode == 'analysis':
        # New-style subcommand
        run_analysis_only(
            samples_path=args.samples,
            config_path=args.config,
            output_dir=args.output_dir,
            prompt_analysis_dir=args.analysis_dir
        )
    elif args.mode == 'script' or hasattr(args, 'script'):
        # Script mode (either new-style or old-style)
        script = args.script
        
        # Resolve the script path
        if not os.path.isabs(script):
            script_path = os.path.join(PROJECT_ROOT, "scripts", script)
            # If script doesn't have .py extension, add it
            if not script_path.endswith('.py'):
                script_path += '.py'
        else:
            script_path = script
        
        # Check if the script exists
        if not os.path.exists(script_path):
            print(f"Error: Script {script_path} not found")
            sys.exit(1)
        
        # Get relative path from PROJECT_ROOT for SLURM script
        rel_script_path = os.path.relpath(script_path, PROJECT_ROOT)
        
        # Run locally or submit to SLURM
        if args.slurm:
            submit_slurm_job(
                rel_script_path, 
                remaining_args,
                job_name=args.job_name,
                mem_per_cpu=args.mem_per_cpu,
                gpus=args.gpus,
                gpumem=args.gpumem,
                ntasks=args.ntasks,
                time=args.time
            )
        else:
            run_local(script_path, remaining_args, use_scratch=not args.no_scratch)
    else:
        # No mode specified, show help
        parser.print_help()

if __name__ == "__main__":
    main() 