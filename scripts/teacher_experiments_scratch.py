#!/usr/bin/env python3
import os
import sys
import subprocess
import pathlib

# Get the username from environment variables
user = os.environ.get("USER")

# 1) Point Hugging Face caching directories to scratch with expanded username
os.environ["TRANSFORMERS_CACHE"] = f"/cluster/scratch/{user}/model_cache"
os.environ["HF_HOME"] = f"/cluster/scratch/{user}/model_cache"
os.environ["HF_DATASETS_CACHE"] = f"/cluster/scratch/{user}/model_cache/datasets"
os.environ["TORCH_HOME"] = f"/cluster/scratch/{user}/model_cache/torch"

# 2) Identify the actual teacher_experiments.py script
current_dir = pathlib.Path(__file__).parent.absolute()
teacher_script = current_dir / "teacher_experiments.py"

# 3) Forward command line arguments to teacher_experiments.py
cmd = [sys.executable, str(teacher_script)] + sys.argv[1:]
print("Launching teacher_experiments.py with scratch-based caching:")
print(f"TRANSFORMERS_CACHE={os.environ['TRANSFORMERS_CACHE']}")
subprocess.run(cmd, check=True)

