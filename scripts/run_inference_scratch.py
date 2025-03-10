#!/usr/bin/env python3
import os
import sys
import subprocess
import pathlib

# Set Hugging Face cache directory to scratch space
os.environ["TRANSFORMERS_CACHE"] = "/cluster/scratch/sstorf/model_cache"
os.environ["HF_HOME"] = "/cluster/scratch/sstorf/model_cache"
os.environ["HF_DATASETS_CACHE"] = "/cluster/scratch/sstorf/model_cache/datasets"
os.environ["TORCH_HOME"] = "/cluster/scratch/sstorf/model_cache/torch"

# Get the current directory
current_dir = pathlib.Path(__file__).parent.absolute()

# Construct the path to the original inference.py script
inference_script = current_dir / "inference.py"

# Forward all command line arguments to the original script
cmd = [sys.executable, str(inference_script)] + sys.argv[1:]

# Run the inference script with the environment variables set
print("Starting inference with model cache in scratch space...")
print(f"Cache directory: {os.environ['TRANSFORMERS_CACHE']}")
subprocess.run(cmd) 