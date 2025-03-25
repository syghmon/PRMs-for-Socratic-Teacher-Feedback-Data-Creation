import torch.multiprocessing as mp
# Set multiprocessing start method to 'spawn' to avoid CUDA reinitialization issues
# This must be done before any CUDA operations
mp.set_start_method('spawn', force=True)

import json
import argparse
import os
import pathlib
import sys
import gc
import torch
import torch.distributed  # Add explicit import for torch.distributed
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from transformers import AutoTokenizer
from requests.exceptions import ConnectionError
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

from utils.math_eval import MathEvaluator
from utils.evaluation_utils import evaluate_predictions

# Set PyTorch CUDA memory allocation mode to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face token from environment variables and set it for the libraries
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["HF_TOKEN"] = hf_token
    logger.info("Hugging Face token loaded from .env file")
else:
    logger.warning("HUGGINGFACE_TOKEN not found in .env file")

# Get the project root directory (parent of scripts)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()

# -------------------------------------------------------------------------
# System prompts for teacher and student (SmolLM2 chat style)
# -------------------------------------------------------------------------
TEACHER_SYSTEM_PROMPT = (
    "You are a reliable math teacher who only provides short hints. "
    "You are given a problem and its solution. "
)

STUDENT_SYSTEM_PROMPT = (
    "You are a helpful math tutor.\nFor each problem, reason step by step with a numbered list and put your final answer in \\boxed{}."
)

# -------------------------------------------------------------------------
# Answer extraction functions
# -------------------------------------------------------------------------
def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Extracts the last boxed expression from a string.
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None

def remove_boxed(s: str) -> Optional[str]:
    """
    Removes the \boxed or \fbox formatting from a string.
    """
    if s is None:
        return None
        
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    elif "\\boxed{" in s:
        left = "\\boxed{"
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    elif "\\fbox{" in s:
        left = "\\fbox{"
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    else:
        return s

def get_answer_expr(answer: str) -> str:
    """
    Extracts the mathematical expression from the answer.
    Only uses boxed expressions.
    """
    try:
        extracted = remove_boxed(last_boxed_only_string(answer))
        return extracted if extracted else ""
    except Exception:
        # Fall back to last line if no boxed expression found
        return answer.split("\n")[-1]

def parse_hint_markdown_bold_colon(response_text: str) -> str:
    """
    Extracts hints formatted as "**HINT**: <text>" from teacher response.
    """
    lines = response_text.splitlines()
    in_hint = False
    collected = []
    for line in lines:
        if not in_hint and re.search(r'^\s*\*\*hint\*\*:', line, re.IGNORECASE):
            in_hint = True
            extracted = re.sub(r'^\s*\*\*hint\*\*:\s*', '', line, flags=re.IGNORECASE)
            collected.append(extracted)
        elif in_hint:
            if line.strip() == '':
                break
            collected.append(line)
    if collected:
        return "\n".join(collected).strip()
    return "Warning: No recognized bold-colon hint found."

# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------
def safe_model_id(model_name: str) -> str:
    """
    Convert a model name into a safe tag for use in filenames.
    """
    base = model_name.split("/")[-1]
    base = base.replace(" ", "_").replace(":", "_").replace("\\", "_").replace("/", "_")
    return base

def enhanced_cleanup_memory(*objects: Any) -> None:
    """
    More aggressive memory cleanup for large models.
    """
    # Delete objects passed in
    for obj in objects:
        del obj
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    # More aggressive memory management
    if torch.cuda.is_available():
        # Get current device
        device = torch.cuda.current_device()
        # Force synchronization to ensure all operations are complete
        torch.cuda.synchronize(device)
        # Empty cache after sync
        torch.cuda.empty_cache()
    
    logger.debug("Enhanced memory cleanup completed")

def validate_config(config: Dict) -> bool:
    """
    Validate the configuration file has all required fields.
    """
    required_fields = ["teacher_model", "teacher_prompts", "student_models"]
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required field in config: {field}")
            return False
    
    # Validate teacher prompts
    for i, prompt in enumerate(config["teacher_prompts"]):
        if "name" not in prompt or "prompt" not in prompt:
            logger.error(f"Teacher prompt at index {i} missing required fields (name, prompt)")
            return False
    
    # Validate student models
    for i, model in enumerate(config["student_models"]):
        if "model_name" not in model:
            logger.error(f"Student model at index {i} missing required 'model_name' field")
            return False
    
    return True

def tokenize_prompt(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str) -> List[int]:
    """
    Tokenize a prompt with proper error handling and fallbacks.
    """
    # First try silently checking if the tokenizer has a chat template
    has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    
    # If no chat template, just combine the prompts directly without logging warnings
    if not has_chat_template:
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        return tokenizer.encode(combined_prompt)
        
    # Try with system role first, but don't log warnings if it fails
    try:
        token_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            add_bos=True,
            add_generation_prompt=True,
            tokenize=True
        )
        return token_ids
    except Exception:
        # Silently fall back to user role only with combined content
        try:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            token_ids = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": combined_prompt}
                ],
                add_bos=True,
                add_generation_prompt=True,
                tokenize=True
            )
            return token_ids
        except Exception:
            # Final fallback - just tokenize directly without logging
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            return tokenizer.encode(combined_prompt)
    
    return token_ids

# -------------------------------------------------------------------------
# Model loading functions
# -------------------------------------------------------------------------
def load_model(model_name: str, tensor_parallel_size: Optional[int] = None) -> LLM:
    """
    Load a model with vLLM from Hugging Face.
    """
    logger.info(f"Loading model: {model_name}")
    try:
        # Check for tensor parallelism settings from environment or arguments
        env_tp_size = os.environ.get("VLLM_TP_SIZE")
        max_model_len = os.environ.get("VLLM_MAX_MODEL_LEN")
        
        # Prepare vLLM arguments
        llm_kwargs = {
            "model": model_name,
            "trust_remote_code": True,
            "disable_custom_all_reduce": True,  # Use standard PyTorch all-reduce
            "enforce_eager": True,  # Disable CUDA graphs for better stability
            "gpu_memory_utilization": 0.85  # Slightly reduce memory utilization for stability
        }
        
        # Add tensor parallelism if specified - prioritize function argument over env var
        if tensor_parallel_size is not None:
            logger.info(f"Using tensor parallelism with tp_size={tensor_parallel_size} (from config)")
            llm_kwargs["tensor_parallel_size"] = tensor_parallel_size
        elif env_tp_size:
            logger.info(f"Using tensor parallelism with tp_size={env_tp_size} (from environment)")
            llm_kwargs["tensor_parallel_size"] = int(env_tp_size)
        
        # Add max model length if specified
        if max_model_len:
            logger.info(f"Using max_model_len={max_model_len}")
            llm_kwargs["max_model_len"] = int(max_model_len)
            
        # vLLM will use HF_TOKEN automatically for authentication
        llm = LLM(**llm_kwargs)
        
        # Wait for workers to be ready
        logger.info("Waiting for vLLM workers to initialize...")
        import time
        time.sleep(2)  # Give workers time to initialize
        
        return llm
    except ConnectionError as e:
        logger.error(f"Connection error downloading/loading model '{model_name}': {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Could not load model '{model_name}': {e}")
        sys.exit(1)

def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load the corresponding tokenizer for a given model.
    """
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

# -------------------------------------------------------------------------
# Core functions for teacher and student inference
# -------------------------------------------------------------------------
def generate_teacher_hints(
    teacher_llm: LLM, 
    teacher_tokenizer: AutoTokenizer,
    samples: List[Dict], 
    prompt_config: Dict
) -> List[str]:
    """
    Generate hints using an already loaded teacher model.
    """
    prompt_name = prompt_config["name"]
    teacher_hint_prompt = prompt_config["prompt"]
    teacher_temperature = prompt_config.get("temperature", 0.7)
    teacher_top_p = prompt_config.get("top_p", 0.9)
    teacher_max_tokens = prompt_config.get("max_tokens", 256)
    
    # Prepare prompts
    logger.info(f"Preparing teacher prompt templates for {len(samples)} problems...")
    prompts = []
    
    for item in samples:
        problem_text = item["problem"]
        solution_text = item.get("solution", item.get("answer", ""))
        
        # Create a plain text prompt instead of message structure
        prompt_text = f"{TEACHER_SYSTEM_PROMPT}\n\n{teacher_hint_prompt}\n\nProblem:\n{problem_text}\n\nSolution:\n{solution_text}"
        prompts.append(prompt_text)
    
    # Set up sampling parameters
    teacher_sampling_params = SamplingParams(
        temperature=teacher_temperature,
        top_p=teacher_top_p,
        max_tokens=teacher_max_tokens
    )
    
    # Generate hints
    logger.info(f"Generating teacher hints for {len(samples)} problems...")
    teacher_outputs = teacher_llm.generate(
        prompts=prompts,
        sampling_params=teacher_sampling_params
    )
    
    # Extract hints from model outputs
    hints = []
    for output_obj in teacher_outputs:
        raw_text = output_obj.outputs[0].text.strip()
        hint_text = parse_hint_markdown_bold_colon(raw_text)
        hints.append(hint_text)
    
    # Clean up intermediate objects but not the model
    enhanced_cleanup_memory(teacher_outputs)
    
    return hints

def prepare_no_hint_prompts(samples: List[Dict], num_samples: int) -> Tuple[List[str], Dict[int, int]]:
    """
    Prepare the student prompts without hints.
    """
    no_hint_prompts = []
    prompt_id_map_no_hint = {}

    for idx, item in enumerate(samples):
        problem_text = item["problem"]
        user_content = (
            f"Reason step by step with a numbered list and put your final answer in \\boxed{{}}.\n\n"
            f"**Example**:\n\n"
            f"If the linear function is y=2x-3, and it is shifted 3 units upwards, what is the new equation?\n"
            f"Step 1: The original function is y=2x-3.\n"
            f"Step 2: When a function is shifted upwards by 3 units, we add 3 to the right side of the equation.\n"
            f"Step 3: y=2x-3+3.\n"
            f"Step 4: Simplifying, y=2x.\n\n"
            f"\\boxed{{y=2x}}.\n\n"
            f"Now solve this problem:\n{problem_text}\n\n"
        )
        # For each problem, create num_samples identical prompts (for sampling diversity)
        for _ in range(num_samples):
            no_hint_prompts.append(user_content)
            # Track which problem index each prompt corresponds to
            prompt_id_map_no_hint[len(no_hint_prompts) - 1] = idx

    return no_hint_prompts, prompt_id_map_no_hint

def prepare_with_hint_prompts(samples: List[Dict], hints: List[str], num_samples: int) -> Tuple[List[str], Dict[int, int]]:
    """
    Prepare the student prompts with hints.
    """
    with_hint_prompts = []
    prompt_id_map_with_hint = {}
    
    for idx, item in enumerate(samples):
        problem_text = item["problem"]
        hint_text = hints[idx]
        user_content = (
            f"Reason step by step with a numbered list and put your final answer in \\boxed{{}}.\n\n"
            f"**Example**:\n\n"
            f"If the linear function is y=2x-3, and it is shifted 3 units upwards, what is the new equation?\n"
            f"Hint:\nAdd 3 to one side of the equation.\n"
            f"Step 1: The original function is y=2x-3.\n"
            f"Step 2: When a function is shifted upwards by 3 units, we add 3 to the right side of the equation.\n"
            f"Step 3: y=2x-3+3.\n"
            f"Step 4: Simplifying, y=2x.\n\n"
            f"\\boxed{{y=2x}}.\n\n"
            f"Now solve this problem:\n{problem_text}\n"
            f"Hint:\n{hint_text}\n\n"
        )
        # For each problem, create num_samples identical prompts (for sampling diversity)
        for _ in range(num_samples):
            with_hint_prompts.append(user_content)
            # Track which problem index each prompt corresponds to
            prompt_id_map_with_hint[len(with_hint_prompts) - 1] = idx

    return with_hint_prompts, prompt_id_map_with_hint

def run_student_inference(
    student_config: Dict,
    prompts: List[str],
    prompt_map: Dict[int, int], 
    samples: List[Dict],
    hints_list: Optional[List[str]] = None,
    include_hint: bool = False,
    prompt_name: Optional[str] = None,
    evaluator: Optional[MathEvaluator] = None,
    use_judge: bool = False,
    reuse_model: Tuple[Optional[LLM], Optional[AutoTokenizer]] = (None, None),
    tensor_parallel_size: Optional[int] = None
) -> Dict:
    """
    Run inference with a student model on a set of prompts.
    """
    model_name = student_config["model_name"]
    temperature = student_config.get("temperature", 0.6)
    top_p = student_config.get("top_p", 0.95)
    max_tokens = student_config.get("max_tokens", 1024)
    student_tag = safe_model_id(model_name)
    
    # Either use provided model/tokenizer or load new ones
    student_llm, student_tokenizer = reuse_model
    if student_llm is None or student_tokenizer is None:
        logger.info(f"\n=== Student Model: {model_name} ===")
        student_llm = load_model(model_name, tensor_parallel_size)
        student_tokenizer = load_tokenizer(model_name)
    
    hint_status = "WITH HINT" if include_hint else "NO HINT"
    prompt_info = f"from {prompt_name}" if prompt_name else ""
    logger.info(f"[{student_tag}] Generating answers ({hint_status} {prompt_info})...")
    
    # Make sampling params from the student's config
    student_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    # Convert raw prompts to plain text format with system prompt
    formatted_prompts = []
    for prompt_text in prompts:
        # Combine system prompt and user content directly
        full_prompt = f"{STUDENT_SYSTEM_PROMPT}\n\n{prompt_text}"
        formatted_prompts.append(full_prompt)

    # Run model inference on formatted prompts
    outputs = student_llm.generate(
        prompts=formatted_prompts,
        sampling_params=student_params
    )

    # Organize results and evaluate
    results = build_hf_results(
        samples,
        outputs,
        prompt_map,
        hints_list=hints_list,
        include_hint=include_hint,
        prompt_name=prompt_name
    )
    
    results = evaluate_predictions(
        results,
        use_judge=use_judge,
        evaluator=evaluator
    )
    
    # Return the model and tokenizer to be reused or cleaned up by caller
    return results, student_llm, student_tokenizer, outputs, None  # No more tokenized_prompts

def build_hf_results(
    samples: List[Dict],
    outputs: List[Any],
    prompt_map: Dict[int, int],
    hints_list: Optional[List[str]] = None,
    include_hint: bool = False,
    prompt_name: Optional[str] = None
) -> List[Dict]:
    """
    Build a huggingface-style results list.
    """
    results = []
    num_problems = len(samples)

    # responses_map will track problem_id -> list of strings
    responses_map = {i: [] for i in range(num_problems)}
    # Track extraction success (whether a boxed expression was found)
    extraction_success_map = {i: [] for i in range(num_problems)}

    # Each item in `outputs` is a vLLM generation result for a single prompt.
    for out_idx, output_obj in enumerate(outputs):
        problem_idx = prompt_map[out_idx]
        # The first (and typically only) text output for each prompt
        text = output_obj.outputs[0].text
        responses_map[problem_idx].append(text)
        
        # Add extraction success flag (whether a boxed expression was found)
        boxed_expr = last_boxed_only_string(text)
        extraction_success = boxed_expr is not None
        extraction_success_map[problem_idx].append(extraction_success)

    # Now build the final structure
    for i, item in enumerate(samples):
        entry = {
            "problem": item.get("problem"),
            "final_answer": item.get("answer"),
            "responses": responses_map[i],
            "extraction_success": extraction_success_map[i]  # Add extraction success flags
        }
        if include_hint and hints_list is not None:
            entry["teacher_hint"] = hints_list[i]
            
        if prompt_name is not None:
            entry["prompt_name"] = prompt_name

        # Copy over any metadata you want to preserve
        if "difficulty_bin" in item:
            entry["difficulty_bin"] = item["difficulty_bin"]
        if "llama8b_solve_rate" in item:
            entry["llama8b_solve_rate"] = item["llama8b_solve_rate"]

        results.append(entry)
    return results

def save_results(
    results: List[Dict],
    base_output_file: str,
    student_tag: str,
    prompt_name: Optional[str] = None,
    with_hint: bool = False
) -> str:
    """
    Save results to a JSON file with appropriate naming.
    """
    base_name, ext = os.path.splitext(base_output_file)
    
    if with_hint:
        # Standardize the naming convention for easier discovery by analysis tool
        output_file = f"{base_name}_{student_tag}_{prompt_name}_with_hint{ext}"
    else:
        output_file = f"{base_name}_{student_tag}_no_hint{ext}"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    hint_status = "WITH-HINT" if with_hint else "NO-HINT"
    prompt_info = f"{prompt_name} " if prompt_name else ""
    logger.info(f"[{student_tag}] Saved {prompt_info}{hint_status} results to {output_file}")
    
    return output_file

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments with detailed help information
    """
    parser = argparse.ArgumentParser(
        description="Teacher–Student inference on math problems, with all config from a JSON file."
    )
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        default="data/samples.json",
        help="Path to the JSON file of sampled problems."
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="results/answers.json",
        help="Base path for output files. Script appends model tags and prompt names."
    )
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        default="scripts/config.json",
        help="Path to the JSON config file defining teacher and student parameters."
    )
    parser.add_argument(
        "--use_judge", 
        action="store_true",
        help="If set, use an LLM judge fallback for correctness checks when standard methods fail."
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Set the logging level"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode (equivalent to --log_level=DEBUG)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Number of GPUs to use for tensor parallelism in vLLM (default: 1)"
    )
    
    args = parser.parse_args()

    # If debug flag is set, override log_level
    if args.debug:
        args.log_level = "DEBUG"
        
    return args

def load_input_data(args: argparse.Namespace) -> Tuple[List[Dict], Dict, str, Optional[int]]:
    """
    Load input data from files specified in arguments
    """
    # Set logging level from command line argument
    logger.setLevel(getattr(logging, args.log_level))
    
    # Load input problems
    input_file = args.input
    if not os.path.isabs(input_file):
        input_file = os.path.join(PROJECT_ROOT, input_file)
    
    logger.info(f"Loading problems from {input_file}")
    with open(input_file, "r") as f:
        samples = json.load(f)
    logger.info(f"Loaded {len(samples)} problems")

    # Create output directory if needed
    base_output_file = args.output
    if not os.path.isabs(base_output_file):
        base_output_file = os.path.join(PROJECT_ROOT, base_output_file)
    os.makedirs(os.path.dirname(base_output_file), exist_ok=True)

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    if not os.path.exists(config_path):
        logger.error(f"Config file does not exist: {config_path}")
        sys.exit(1)

    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as cf:
        config = json.load(cf)
    
    # Validate config
    if not validate_config(config):
        logger.error("Invalid configuration, exiting")
        sys.exit(1)
        
    # Check if tensor_parallel_size is defined in the config and not overridden by command line
    tensor_parallel_size = args.tensor_parallel_size
    if tensor_parallel_size is None and "tensor_parallel_size" in config:
        tensor_parallel_size = config["tensor_parallel_size"]
        logger.info(f"Using tensor_parallel_size={tensor_parallel_size} from config file")
    
    return samples, config, base_output_file, tensor_parallel_size

def main():
    """
    Main entry point for the teacher-student math inference script.
    """
    args = parse_arguments()
    samples, config, base_output_file, tensor_parallel_size = load_input_data(args)
    
    teacher_model_name = config["teacher_model"]
    teacher_prompts = config["teacher_prompts"]
    num_samples = config.get("num_samples", 1)
    student_configs = config["student_models"]
    
    # Optionally, if user wants LLM-based judge, build the evaluator
    evaluator = None
    if args.use_judge:
        logger.info("Initializing LLM judge for correctness evaluation")
        evaluator = MathEvaluator()
    
    # Generate all hints first using the teacher model
    all_hints = {}
    
    # Load teacher model and tokenizer just once for all prompts
    logger.info(f"Loading teacher model: {teacher_model_name}")
    teacher_llm = load_model(teacher_model_name, tensor_parallel_size)
    teacher_tokenizer = load_tokenizer(teacher_model_name)
    
    # For each teacher prompt, generate hints
    for prompt_idx, prompt_config in enumerate(teacher_prompts):
        prompt_name = prompt_config["name"]
        logger.info(f"Processing teacher prompt: {prompt_name} ({prompt_idx+1}/{len(teacher_prompts)})")
        
        # Generate teacher hints using the loaded model
        hints = generate_teacher_hints(
            teacher_llm=teacher_llm, 
            teacher_tokenizer=teacher_tokenizer,
            samples=samples, 
            prompt_config=prompt_config
        )
        all_hints[prompt_name] = hints
    
    # Clean up the teacher model after generating all hints
    logger.info("All hints generated. Unloading teacher model to free memory.")
    enhanced_cleanup_memory(teacher_llm, teacher_tokenizer)
    teacher_llm = None
    teacher_tokenizer = None
    
    # Keep track of whether no-hint results have been saved for each student model
    no_hint_saved = {safe_model_id(model["model_name"]): False for model in student_configs}

    # Now process student models with the pre-generated hints
    for prompt_idx, prompt_config in enumerate(teacher_prompts):
        prompt_name = prompt_config["name"]
        hints = all_hints[prompt_name]
        
        # Prepare no-hint prompts (only for the first teacher prompt)
        no_hint_prompts = []
        prompt_id_map_no_hint = {}
        if prompt_idx == 0:
            no_hint_prompts, prompt_id_map_no_hint = prepare_no_hint_prompts(samples, num_samples)
        
        # Prepare with-hint prompts for the current teacher prompt
        with_hint_prompts, prompt_id_map_with_hint = prepare_with_hint_prompts(samples, hints, num_samples)

        # Run student models one at a time to avoid memory issues
        for stu_idx, student_config in enumerate(student_configs):
            model_name = student_config["model_name"]
            student_tag = safe_model_id(model_name)
            logger.info(f"Processing student model: {model_name} ({stu_idx+1}/{len(student_configs)})")
            
            # Variables to potentially reuse models across inference runs
            student_llm = None
            student_tokenizer = None

            # --- NO-HINT Inference (only for the first teacher prompt) ---
            if prompt_idx == 0 and not no_hint_saved[student_tag]:
                # Run no-hint inference
                results_no_hint, student_llm, student_tokenizer, outputs_no_hint, _ = run_student_inference(
                    student_config=student_config,
                    prompts=no_hint_prompts,
                    prompt_map=prompt_id_map_no_hint,
                    samples=samples,
                    include_hint=False,
                    evaluator=evaluator,
                    use_judge=args.use_judge,
                    reuse_model=(None, None),  # Don't reuse model for the first run
                    tensor_parallel_size=tensor_parallel_size  # Pass tensor_parallel_size
                )
                
                # Save no-hint results
                save_results(results_no_hint, base_output_file, student_tag, with_hint=False)
                
                # Mark this student model as having saved its no-hint results
                no_hint_saved[student_tag] = True
                
                # Clean up outputs but keep model and tokenizer for reuse
                enhanced_cleanup_memory(results_no_hint, outputs_no_hint)

            # --- WITH-HINT Inference ---
            # Run with-hint inference, potentially reusing the model from no-hint run
            results_with_hint, student_llm, student_tokenizer, outputs_with_hint, _ = run_student_inference(
                student_config=student_config,
                prompts=with_hint_prompts,
                prompt_map=prompt_id_map_with_hint,
                samples=samples,
                hints_list=hints,
                include_hint=True,
                prompt_name=prompt_name,
                evaluator=evaluator,
                use_judge=args.use_judge,
                reuse_model=(student_llm, student_tokenizer),
                tensor_parallel_size=tensor_parallel_size  # Pass tensor_parallel_size
            )
            
            # Save with-hint results
            save_results(results_with_hint, base_output_file, student_tag, prompt_name, with_hint=True)
            
            # Clean up everything including model and tokenizer
            enhanced_cleanup_memory(
                student_llm, student_tokenizer, 
                results_with_hint, outputs_with_hint
            )
                
    logger.info("\nAll models and prompts completed. Exiting.")
    
    # Cleanup PyTorch distributed processes to avoid the NCCL warning
    if torch.distributed.is_initialized():
        logger.info("Cleaning up PyTorch distributed process groups")
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()