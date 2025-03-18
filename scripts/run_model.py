#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import pathlib
import tempfile
import json
import logging
from typing import List, Optional, Dict, Any, Union

# Get the project root directory
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_model")

# Constants
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config.json")
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_ANALYSIS_DIR = "prompt_analysis"
DEFAULT_SCRATCH_DIR = "/cluster/scratch/sstorf/model_cache"

###############################################################################
# Path Handling Utilities
###############################################################################

def ensure_absolute_path(path: str, base_dir: pathlib.Path = PROJECT_ROOT) -> str:
    """Convert a relative path to an absolute path if necessary."""
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)

def ensure_dir_exists(path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)

def resolve_script_path(script_name: str) -> str:
    """Resolve a script name to its absolute path."""
    if os.path.isabs(script_name):
        script_path = script_name
    else:
        script_path = os.path.join(PROJECT_ROOT, "scripts", script_name)
        # Add .py extension if not present
        if not script_path.endswith('.py'):
            script_path += '.py'
    
    # Verify the script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    return script_path

###############################################################################
# Configuration Management
###############################################################################

class ConfigManager:
    """Manages configuration loading and validation."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load a configuration file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    @staticmethod
    def get_student_models(config: Dict[str, Any]) -> List[str]:
        """Extract student model IDs from config."""
        return [ConfigManager.safe_model_id(model["model_name"]) 
                for model in config.get("student_models", [])]
    
    @staticmethod
    def get_prompt_names(config: Dict[str, Any]) -> List[str]:
        """Extract prompt names from config."""
        return [prompt["name"] for prompt in config.get("teacher_prompts", [])]
    
    @staticmethod
    def safe_model_id(model_name: str) -> str:
        """
        Convert a model name to a safe tag for use in filenames.
        Example: 'huggingfaceTB/SmolLM2-1.7B-Instruct' -> 'SmolLM2-1.7B-Instruct'
        """
        base = model_name.split("/")[-1]
        # Replace characters that might cause filename issues
        return base.replace(" ", "_").replace(":", "_").replace("\\", "_").replace("/", "_")

###############################################################################
# Environment Setup
###############################################################################

class EnvironmentManager:
    """Manages environment setup for model execution."""
    
    @staticmethod
    def setup_env_vars(use_scratch: bool = True, scratch_dir: str = DEFAULT_SCRATCH_DIR) -> Dict[str, str]:
    """Set up environment variables for model caching."""
        env_vars = {}
        
    if use_scratch:
            env_vars["TRANSFORMERS_CACHE"] = scratch_dir
            env_vars["HF_HOME"] = scratch_dir
            env_vars["HF_DATASETS_CACHE"] = os.path.join(scratch_dir, "datasets")
            env_vars["TORCH_HOME"] = os.path.join(scratch_dir, "torch")
            
            # Also update the current process environment
            os.environ.update(env_vars)
            logger.info(f"Using scratch directory for model cache: {scratch_dir}")
        
        return env_vars

    @staticmethod
    def get_env_setup_script(scratch_dir: str = DEFAULT_SCRATCH_DIR) -> str:
        """Get bash script lines to set up the environment."""
        return "\n".join([
            f"export TRANSFORMERS_CACHE=\"{scratch_dir}\"",
            f"export HF_HOME=\"{scratch_dir}\"",
            f"export HF_DATASETS_CACHE=\"{scratch_dir}/datasets\"",
            f"export TORCH_HOME=\"{scratch_dir}/torch\""
        ])

###############################################################################
# Execution Runners
###############################################################################

class Runner:
    """Base class for execution runners."""
    
    def run(self, script_path: str, script_args: List[str]) -> None:
        """Run a script with the given arguments."""
        raise NotImplementedError("Subclasses must implement run()")

class LocalRunner(Runner):
    """Runs scripts locally."""
    
    def __init__(self, use_scratch: bool = True):
        self.use_scratch = use_scratch
    
    def run(self, script_path: str, script_args: List[str]) -> None:
        """Run a script locally with the correct environment."""
        # Set up environment variables
        EnvironmentManager.setup_env_vars(self.use_scratch)
    
    # Construct the command
    cmd = [sys.executable, script_path] + script_args
        logger.info(f"Running locally: {' '.join(cmd)}")
    
    # Execute the command
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}")
            raise

class SlurmRunner(Runner):
    """Submits scripts to SLURM cluster."""
    
    def __init__(self, 
                       job_name: str = "model_job",
                       mem_per_cpu: str = "16G",
                       gpus: int = 1,
                       gpumem: str = "40g",
                       ntasks: int = 4,
                time: str = "2:00:00",
                scratch_dir: str = DEFAULT_SCRATCH_DIR):
        self.job_name = job_name
        self.mem_per_cpu = mem_per_cpu
        self.gpus = gpus
        self.gpumem = gpumem
        self.ntasks = ntasks
        self.time = time
        self.scratch_dir = scratch_dir
    
    def run(self, script_path: str, script_args: List[str]) -> None:
        """Submit a job to SLURM."""
        # Get the relative script path for the SLURM script
        rel_script_path = os.path.relpath(script_path, PROJECT_ROOT)
        
        # Create the SLURM script
        slurm_script = self._create_slurm_script(rel_script_path, script_args)
        
        # Submit the job
        cmd = ["sbatch", slurm_script]
        logger.info(f"Submitting SLURM job: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Job submitted successfully: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Job submission failed: {e.stderr}")
            raise
        finally:
            # Clean up the temporary file
            os.unlink(slurm_script)
    
    def _create_slurm_script(self, script_path: str, script_args: List[str]) -> str:
        """Create a temporary SLURM script file."""
        # Ensure logs directory exists
        logs_dir = os.path.join(PROJECT_ROOT, "logs")
        ensure_dir_exists(logs_dir)
        
        # Get environment setup commands
        env_vars_str = EnvironmentManager.get_env_setup_script(self.scratch_dir)
    
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={self.job_name}
#SBATCH --output=logs/{self.job_name}_%j.out
#SBATCH --error=logs/{self.job_name}_%j.err
#SBATCH --ntasks={self.ntasks}
#SBATCH --mem-per-cpu={self.mem_per_cpu}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={self.gpus}
#SBATCH --gres=gpumem:{self.gpumem}
#SBATCH --time={self.time}

module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

# Activate your Python virtual environment
source ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation/socratic-env/bin/activate

cd ~/PRMs-for-Socratic-Teacher-Feedback-Data-Creation

echo "Starting job: {self.job_name}"
echo "Job started at $(date)"

# Print GPU information
echo "======== GPU INFORMATION ========"
nvidia-smi
echo ""

# Check GPU count
echo "======== GPU COUNT CHECK ========"
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Number of visible GPUs: $NUM_GPUS"
if [ $NUM_GPUS -lt {self.gpus} ]; then
    echo "WARNING: Requested {self.gpus} GPUs but only $NUM_GPUS are visible!"
fi
echo ""

# Set cache directories
{env_vars_str}

# Run the script
python {script_path} {' '.join(script_args)}

echo "Job completed at $(date)"

deactivate
"""
    
    # Create a temporary file for the SLURM script
    fd, temp_path = tempfile.mkstemp(suffix='.sbatch')
    with os.fdopen(fd, 'w') as f:
        f.write(slurm_content)
    
    return temp_path

###############################################################################
# Pipeline Components
###############################################################################

class Pipeline:
    """Base class for execution pipelines."""
    
    def run(self) -> None:
        """Run the pipeline."""
        raise NotImplementedError("Subclasses must implement run()")

class PromptComparisonPipeline(Pipeline):
    """Pipeline for running prompt comparisons."""
    
    def __init__(self,
    samples_path: str,
    config_path: Optional[str] = None,
                output_dir: str = DEFAULT_OUTPUT_DIR,
                analysis_dir: str = DEFAULT_ANALYSIS_DIR,
    use_judge: bool = False,
                runner: Runner = None):
        # Set up paths
        self.samples_path = ensure_absolute_path(samples_path)
        self.config_path = ensure_absolute_path(config_path or DEFAULT_CONFIG_PATH)
        self.output_dir = ensure_absolute_path(output_dir)
        self.analysis_dir = ensure_absolute_path(analysis_dir)
        
        # Set up options
        self.use_judge = use_judge
        self.runner = runner or LocalRunner()
    
    # Create output directories
        ensure_dir_exists(self.output_dir)
        ensure_dir_exists(self.analysis_dir)
        
        # Load configuration
        self.config = ConfigManager.load_config(self.config_path)
        self.student_models = ConfigManager.get_student_models(self.config)
        self.prompt_names = ConfigManager.get_prompt_names(self.config)
    
    def run(self) -> None:
        """Run the complete pipeline."""
        self._run_inference()
        self._run_analysis()
    
    def _run_inference(self) -> None:
        """Run the inference step."""
        logger.info("=== Running Inference with Multiple Teacher Prompts ===")
        
        # Set up inference arguments
        inference_script = resolve_script_path("inference.py")
    inference_args = [
            "--input", self.samples_path,
            "--output", os.path.join(self.output_dir, "answers.json"),
            "--config", self.config_path
        ]
        
        if self.use_judge:
            inference_args.append("--use_judge")
        
        # Run inference
        self.runner.run(inference_script, inference_args)
    
    def _run_analysis(self) -> None:
        """Run the analysis step."""
        logger.info("=== Running Prompt Analysis ===")
        
        # Set up analysis arguments
        analysis_script = resolve_script_path("prompt_analysis.py")
    analysis_args = [
            "--samples", self.samples_path,
            "--results_dir", self.output_dir,
            "--output_dir", self.analysis_dir,
            "--student_models", ",".join(self.student_models),
            "--prompt_names", ",".join(self.prompt_names)
    ]
    
    # Run analysis
        self.runner.run(analysis_script, analysis_args)
        
        logger.info("=== Pipeline Complete ===")
        logger.info(f"Inference results saved to: {self.output_dir}")
        logger.info(f"Analysis results saved to: {self.analysis_dir}")

###############################################################################
# Command Line Interface
###############################################################################

def parse_script_args(subparsers):
    """Define arguments for the 'script' subcommand."""
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
    
def parse_pipeline_args(subparsers):
    """Define arguments for the 'pipeline' subcommand."""
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the complete prompt comparison pipeline')
    pipeline_parser.add_argument("--samples", "-s", type=str, required=True,
                        help="Path to the samples JSON file")
    pipeline_parser.add_argument("--config", "-c", type=str, 
                        help="Path to the config JSON file (default: scripts/config.json)")
    pipeline_parser.add_argument("--output-dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save inference results")
    pipeline_parser.add_argument("--analysis-dir", "-a", type=str, default=DEFAULT_ANALYSIS_DIR,
                        help="Directory to save analysis results")
    pipeline_parser.add_argument("--use-judge", action="store_true",
                        help="Use LLM judge for evaluating answers")
    pipeline_parser.add_argument("--slurm", action="store_true",
                        help="Submit inference as a SLURM job")
    pipeline_parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs (for SLURM mode)")
    pipeline_parser.add_argument("--time", type=str, default="8:00:00",
                        help="Maximum job runtime (for SLURM mode)")
    
def parse_analysis_args(subparsers):
    """Define arguments for the 'analysis' subcommand."""
    analysis_parser = subparsers.add_parser('analysis', help='Run only the prompt analysis')
    analysis_parser.add_argument("--samples", "-s", type=str, required=True,
                        help="Path to the samples JSON file")
    analysis_parser.add_argument("--config", "-c", type=str, 
                        help="Path to the config JSON file (default: scripts/config.json)")
    analysis_parser.add_argument("--output-dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory with inference results")
    analysis_parser.add_argument("--analysis-dir", "-a", type=str, default=DEFAULT_ANALYSIS_DIR,
                        help="Directory to save analysis results")
    
def parse_backward_compat_args(parser):
    """Define backward compatibility arguments."""
    parser.add_argument("--run-pipeline", action="store_true",
                    help="[DEPRECATED] Run the complete prompt comparison pipeline")
    parser.add_argument("--run-analysis-only", action="store_true",
                    help="[DEPRECATED] Run only the prompt analysis part")
    
    # These arguments are used with --run-pipeline and --run-analysis-only
    parser.add_argument("--samples", type=str,
                        help="Path to the samples JSON file (for pipeline mode)")
    parser.add_argument("--config", type=str,
                        help="Path to the config JSON file (for pipeline mode)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save results (for pipeline mode)")
    parser.add_argument("--analysis-dir", type=str, default=DEFAULT_ANALYSIS_DIR,
                        help="Directory to save analysis results (for pipeline mode)")
    parser.add_argument("--use-judge", action="store_true",
                        help="Use LLM judge for evaluating answers (for pipeline mode)")
    
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified utility for running model scripts locally or on SLURM"
    )
    
    # Define subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Add arguments for each subcommand
    parse_script_args(subparsers)
    parse_pipeline_args(subparsers)
    parse_analysis_args(subparsers)
    
    # Add backward compatibility arguments
    parse_backward_compat_args(parser)
    
    # Parse arguments
    args, remaining_args = parser.parse_known_args()
    
    return args, remaining_args

def main():
    """Main entry point."""
    args, remaining_args = parse_args()
    
    try:
        # Handle backward compatibility modes
    if args.run_pipeline:
            logger.warning("--run-pipeline is deprecated. Use 'pipeline' subcommand instead.")
        if args.samples is None:
                raise ValueError("--samples is required with --run-pipeline")
                
            # Create and run pipeline
            pipeline = PromptComparisonPipeline(
            samples_path=args.samples,
            config_path=args.config,
            output_dir=args.output_dir,
                analysis_dir=args.analysis_dir,
            use_judge=args.use_judge,
                runner=LocalRunner()
        )
            pipeline.run()
            
    elif args.run_analysis_only:
            logger.warning("--run-analysis-only is deprecated. Use 'analysis' subcommand instead.")
        if args.samples is None:
                raise ValueError("--samples is required with --run-analysis-only")
                
            # Load configuration
            config_path = args.config or DEFAULT_CONFIG_PATH
            config = ConfigManager.load_config(ensure_absolute_path(config_path))
            
            # Set up analysis arguments
            analysis_script = resolve_script_path("prompt_analysis.py")
            analysis_args = [
                "--samples", ensure_absolute_path(args.samples),
                "--results_dir", ensure_absolute_path(args.output_dir),
                "--output_dir", ensure_absolute_path(args.analysis_dir),
                "--student_models", ",".join(ConfigManager.get_student_models(config)),
                "--prompt_names", ",".join(ConfigManager.get_prompt_names(config))
            ]
            
            # Run analysis
            LocalRunner().run(analysis_script, analysis_args)
            
        # Handle new-style subcommands
    elif args.mode == 'pipeline':
            # Create runner
            if args.slurm:
                runner = SlurmRunner(
                    job_name="teacher_inference",
            gpus=args.gpus,
            time=args.time
        )
            else:
                runner = LocalRunner()
                
            # Create and run pipeline
            pipeline = PromptComparisonPipeline(
            samples_path=args.samples,
            config_path=args.config,
            output_dir=args.output_dir,
                analysis_dir=args.analysis_dir,
                use_judge=args.use_judge,
                runner=runner
            )
            
            if args.slurm:
                # Only run inference when using SLURM
                pipeline._run_inference()
                logger.info("\nInference job submitted to SLURM. Once it completes, run the analysis step with:")
                logger.info(f"python scripts/run_model.py analysis --samples {args.samples} --config {args.config}")
        else:
                pipeline.run()
                
        elif args.mode == 'analysis':
            # Load configuration
            config_path = args.config or DEFAULT_CONFIG_PATH
            config = ConfigManager.load_config(ensure_absolute_path(config_path))
            
            # Set up analysis arguments
            analysis_script = resolve_script_path("prompt_analysis.py")
            analysis_args = [
                "--samples", ensure_absolute_path(args.samples),
                "--results_dir", ensure_absolute_path(args.output_dir),
                "--output_dir", ensure_absolute_path(args.analysis_dir),
                "--student_models", ",".join(ConfigManager.get_student_models(config)),
                "--prompt_names", ",".join(ConfigManager.get_prompt_names(config))
            ]
            
            # Run analysis
            logger.info("=== Running Prompt Analysis ===")
            LocalRunner().run(analysis_script, analysis_args)
            logger.info("=== Analysis Complete ===")
            logger.info(f"Analysis results saved to: {args.analysis_dir}")
            
        elif args.mode == 'script' or hasattr(args, 'script'):
            # Get the script path
            try:
                script_path = resolve_script_path(args.script)
            except FileNotFoundError as e:
                logger.error(f"Error: {e}")
            sys.exit(1)
        
        # Get relative path from PROJECT_ROOT for SLURM script
        rel_script_path = os.path.relpath(script_path, PROJECT_ROOT)
        
        # Run locally or submit to SLURM
        if args.slurm:
                runner = SlurmRunner(
                job_name=args.job_name,
                mem_per_cpu=args.mem_per_cpu,
                gpus=args.gpus,
                gpumem=args.gpumem,
                ntasks=args.ntasks,
                time=args.time
            )
                runner.run(rel_script_path, remaining_args)
            else:
                runner = LocalRunner(use_scratch=not args.no_scratch)
                runner.run(script_path, remaining_args)
    else:
        # No mode specified, show help
            logger.info("No operation mode specified.")
            import argparse
            parser = argparse.ArgumentParser(
                description="Unified utility for running model scripts locally or on SLURM"
            )
        parser.print_help()
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 