import json
import argparse
import os
import pathlib
import sys
import gc
import torch  # for torch.cuda.empty_cache(), if you're using CUDA
import re  # for regex-based hint extraction
from transformers import AutoTokenizer
from requests.exceptions import ConnectionError
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

from utils.math_eval import MathEvaluator  # optional for LLM-based judge
from utils.evaluation_utils import evaluate_predictions  # single entry point for scoring

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face token from environment variables and set it for the libraries
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    # Set the standard environment variables that HF libraries check for authentication
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["HF_TOKEN"] = hf_token
    print("Hugging Face token loaded from .env file")
else:
    print("Warning: HUGGINGFACE_TOKEN not found in .env file")

# Get the project root directory (parent of scripts)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()

# -------------------------------------------------------------------------
# System prompts for teacher and student (SmolLM2 chat style)
# -------------------------------------------------------------------------
TEACHER_SYSTEM_PROMPT = (
    "You are a reliable math teacher who only provides short Socratic hints. "
    "You are given a problem and a solution."
    "Think about the problem and the solution, and provide a hint for the problem that will help the student solve it. "
    "Always follow the reasoning with exactly one line starting with '**HINT**:'. "
)

STUDENT_SYSTEM_PROMPT = (
    "You are a helpful math tutor.\nFor each problem, reason step by step with a numbered list and put your final answer in \\boxed{}."
)


def parse_hint_markdown_bold_colon(response_text: str) -> str:
    """
    For a bold hint format like:
      **HINT**: Here is the hint text...
      (and possibly more lines)
      Then a blank line or end of text.
    We return the lines collected after '**HINT**:'.
    """
    lines = response_text.splitlines()
    in_hint = False
    collected = []
    for line in lines:
        # Check if the line begins with **HINT**: (case-insensitive)
        if not in_hint and re.search(r'^\s*\*\*hint\*\*:', line, re.IGNORECASE):
            in_hint = True
            # Remove the **HINT**: prefix from this line
            extracted = re.sub(r'^\s*\*\*hint\*\*:\s*', '', line, flags=re.IGNORECASE)
            collected.append(extracted)
        elif in_hint:
            # Stop if we hit a blank line
            if line.strip() == '':
                break
            collected.append(line)
    if collected:
        return "\n".join(collected).strip()
    return "Warning: No recognized bold-colon hint found."


def safe_model_id(model_name: str) -> str:
    """
    Convert a model name (e.g. 'huggingfaceTB/SmolLM2-1.7B-Instruct')
    into a safe tag (e.g. 'SmolLM2-1.7B-Instruct') for use in filenames.
    """
    base = model_name.split("/")[-1]
    # Replace characters that might cause filename issues
    base = base.replace(" ", "_").replace(":", "_").replace("\\", "_").replace("/", "_")
    return base


def load_model(model_name: str) -> LLM:
    """
    Attempt to load a model with vLLM from Hugging Face in (FP16) full precision.
    """
    print(f"Loading model: {model_name}")
    try:
        # The HF_TOKEN environment variable should already be set at the top of the script
        # vLLM will use this automatically for authentication
        llm = LLM(model=model_name, trust_remote_code=True)
        return llm
    except ConnectionError as e:
        print(f"[Error] Connection error downloading/loading model '{model_name}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Could not load model '{model_name}': {e}")
        sys.exit(1)


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load the corresponding tokenizer for a given model.
    """
    # The HF_TOKEN environment variable should already be set at the top of the script
    # Transformers will use HUGGING_FACE_HUB_TOKEN or HF_TOKEN automatically
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)


def build_hf_results(samples, outputs, prompt_map, hints_list=None, include_hint=False, prompt_name=None):
    """
    Build a huggingface-style results list:
    Each problem is one record with:
      - "problem"
      - "final_answer" (ground truth)
      - "responses" (list of model outputs, length = num_samples)
      - optionally "teacher_hint" if include_hint = True
      - optionally "prompt_name" if prompt_name is provided
      - any metadata (e.g. difficulty_bin)
    """
    # We assume len(outputs) == len(samples) * num_samples
    # We'll accumulate model outputs for each problem i
    # in a list of length num_samples
    results = []
    num_problems = len(samples)

    # responses_map will track problem_id -> list of strings
    responses_map = {i: [] for i in range(num_problems)}

    # Each item in `outputs` is a vLLM generation result for a single prompt.
    for out_idx, output_obj in enumerate(outputs):
        problem_idx = prompt_map[out_idx]
        # The first (and typically only) text output for each prompt
        text = output_obj.outputs[0].text
        responses_map[problem_idx].append(text)

    # Now build the final structure
    for i, item in enumerate(samples):
        entry = {
            "problem": item.get("problem"),
            "final_answer": item.get("answer"),
            "responses": responses_map[i]
        }
        if include_hint and hints_list is not None:
            entry["teacher_hint"] = hints_list[i]
            # Optionally, you can keep the full response for debugging
            # entry["teacher_full_response"] = raw_hint
            
        if prompt_name is not None:
            entry["prompt_name"] = prompt_name

        # Copy over any metadata you want to preserve
        if "difficulty_bin" in item:
            entry["difficulty_bin"] = item["difficulty_bin"]
        if "llama8b_solve_rate" in item:
            entry["llama8b_solve_rate"] = item["llama8b_solve_rate"]

        results.append(entry)
    return results


def generate_teacher_hints(teacher_model_name, samples, prompt_config):
    """
    Load the teacher model, generate hints based on the provided prompt config,
    and then unload the model to free memory.
    
    Returns:
        List of hint texts for each sample.
    """
    prompt_name = prompt_config["name"]
    teacher_hint_prompt = prompt_config["prompt"]
    teacher_temperature = prompt_config.get("temperature", 0.7)
    teacher_top_p = prompt_config.get("top_p", 0.9)
    teacher_max_tokens = prompt_config.get("max_tokens", 256)
    
    print(f"\n=== Teacher Prompt: {prompt_name} ===")
    
    # Load teacher model and tokenizer
    print(f"Loading teacher model: {teacher_model_name}")
    teacher_llm = load_model(teacher_model_name)
    teacher_tokenizer = load_tokenizer(teacher_model_name)
    
    # Prepare prompts
    print(f"Preparing teacher prompt templates for {len(samples)} problems...")
    tokenized_teacher_prompts = []
    
    for item in samples:
        problem_text = item["problem"]
        solution_text = item.get("answer", "")
        user_content = f"""
{teacher_hint_prompt}

Problem:
{problem_text}

Solution:
{solution_text}
"""

        token_ids = teacher_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content}
            ],
            add_bos=True,
            add_generation_prompt=True,
            tokenize=True
        )
        tokenized_teacher_prompts.append(token_ids)
    
    # Set up sampling parameters
    teacher_sampling_params = SamplingParams(
        temperature=teacher_temperature,
        top_p=teacher_top_p,
        max_tokens=teacher_max_tokens
    )
    
    # Generate hints
    print(f"Generating teacher hints for {len(samples)} problems...")
    teacher_outputs = teacher_llm.generate(
        prompt_token_ids=tokenized_teacher_prompts,
        sampling_params=teacher_sampling_params
    )
    
    # Extract hints from model outputs
    hints = []
    for output_obj in teacher_outputs:
        raw_text = output_obj.outputs[0].text.strip()
        hint_text = parse_hint_markdown_bold_colon(raw_text)
        hints.append(hint_text)
    
    # Clean up teacher model to free memory
    del teacher_llm
    del teacher_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Teacher model unloaded and memory cleared")
    
    return hints


def main():
    parser = argparse.ArgumentParser(
        description="Teacher–Student inference on math problems, with all config from a JSON file."
    )
    parser.add_argument("--input", "-i", type=str, default="data/samples.json",
                        help="Path to the JSON file of sampled problems.")
    parser.add_argument("--output", "-o", type=str, default="results/answers.json",
                        help="Base path for output files. Script appends model tags and prompt names.")
    parser.add_argument("--config", "-c", type=str, default="scripts/config.json",
                        help="Path to the JSON config file defining teacher and student parameters.")
    parser.add_argument("--use_judge", action="store_true",
                        help="If set, we use an LLM judge fallback for correctness checks.")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1) Load input problems
    # -------------------------------------------------------------------------
    input_file = args.input
    if not os.path.isabs(input_file):
        input_file = os.path.join(PROJECT_ROOT, input_file)
    with open(input_file, "r") as f:
        samples = json.load(f)

    # Create output directory if needed
    base_output_file = args.output
    if not os.path.isabs(base_output_file):
        base_output_file = os.path.join(PROJECT_ROOT, base_output_file)
    os.makedirs(os.path.dirname(base_output_file), exist_ok=True)

    # -------------------------------------------------------------------------
    # 2) Load config
    # -------------------------------------------------------------------------
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    if not os.path.exists(config_path):
        print(f"[Error] Config file does not exist: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as cf:
        cfg = json.load(cf)

    teacher_model_name = cfg["teacher_model"]
    teacher_prompts = cfg["teacher_prompts"]
    num_samples = cfg.get("num_samples", 1)

    student_configs = cfg["student_models"]  # list of dicts w/ model_name, temperature, top_p, etc.

    # Optionally, if user wants LLM-based judge, build the evaluator
    evaluator = None
    if args.use_judge:
        evaluator = MathEvaluator()
        
    # Keep track of whether no-hint results have been saved for each student model
    no_hint_saved = {safe_model_id(model["model_name"]): False for model in student_configs}

    # -------------------------------------------------------------------------
    # For each teacher prompt, generate hints and then run student models
    # -------------------------------------------------------------------------
    for prompt_idx, prompt_config in enumerate(teacher_prompts):
        prompt_name = prompt_config["name"]
        
        # Generate teacher hints for this prompt
        hints = generate_teacher_hints(teacher_model_name, samples, prompt_config)
        
        # Prepare student prompts (no-hint and with-hint)
        # a) NO-HINT prompts (only need to prepare these for the first teacher prompt)
        no_hint_prompts = []
        prompt_id_map_no_hint = {}

        if prompt_idx == 0:  # Only prepare no-hint prompts for the first teacher prompt
            for idx, item in enumerate(samples):
                problem_text = item["problem"]
                user_content = (
                    f"Reason step by step with a numbered list and put your final answer in \\boxed{{}}.\n\n**Example**:\n\nIf the linear function is y=2x-3, and it is shifted 3 units upwards, what is the new equation?\nStep 1: The original function is y=2x-3.\nStep 2: When a function is shifted upwards by 3 units, we add 3 to the right side of the equation.\nStep 3: y=2x-3+3.\nStep 4: Simplifying, y=2x.\n\nAnswer: \\boxed{{y=2x}}.\n\nNow solve this problem:\n{problem_text}\n\n"
                )
                # For each problem, create num_samples identical prompts (for sampling diversity)
                for _ in range(num_samples):
                    no_hint_prompts.append(user_content)
                    # Track which problem index each prompt corresponds to
                    prompt_id_map_no_hint[len(no_hint_prompts) - 1] = idx

        # b) WITH-HINT prompts for the current teacher prompt
        with_hint_prompts = []
        prompt_id_map_with_hint = {}
        
        for idx, item in enumerate(samples):
            problem_text = item["problem"]
            hint_text = hints[idx]
            user_content = (
                f"Reason step by step with a numbered list and put your final answer in \\boxed{{}}.\n\n**Example**:\n\nIf the linear function is y=2x-3, and it is shifted 3 units upwards, what is the new equation?\nHint:\nAdd 3 to one side of the equation.\nStep 1: The original function is y=2x-3.\nStep 2: When a function is shifted upwards by 3 units, we add 3 to the right side of the equation.\nStep 3: y=2x-3+3.\nStep 4: Simplifying, y=2x.\n\nAnswer: \\boxed{{y=2x}}.\n\nNow solve this problem:\n{problem_text}\nHint:\n{hint_text}\n\n"
            )
            # For each problem, create num_samples identical prompts (for sampling diversity)
            for _ in range(num_samples):
                with_hint_prompts.append(user_content)
                # Track which problem index each prompt corresponds to
                prompt_id_map_with_hint[len(with_hint_prompts) - 1] = idx

        # Run student models one at a time to avoid memory issues
        for stu_cfg in student_configs:
            model_name = stu_cfg["model_name"]
            stu_temp = stu_cfg.get("temperature", 0.6)
            stu_top_p = stu_cfg.get("top_p", 0.95)
            stu_max_tokens = stu_cfg.get("max_tokens", 1024)
            student_tag = safe_model_id(model_name)

            # --- NO-HINT Inference (only for the first teacher prompt) ---
            if prompt_idx == 0 and not no_hint_saved[student_tag]:
                print(f"\n=== Student Model: {model_name} ===")
                print(f"[{student_tag}] Generating answers (NO HINT)...")
                
                # Load student model
                student_llm = load_model(model_name)
                student_tokenizer = load_tokenizer(model_name)
                
                # Make sampling params from the student's config
                student_params = SamplingParams(
                    temperature=stu_temp,
                    top_p=stu_top_p,
                    max_tokens=stu_max_tokens
                )
                
                # Convert raw prompts to tokenized chat format
                tokenized_no_hint_prompts = []
                for prompt_text in no_hint_prompts:
                    try:
                        # Try with system role first
                        token_ids = student_tokenizer.apply_chat_template(
                            [
                                {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                                {"role": "user",   "content": prompt_text}
                            ],
                            add_bos=True,
                            add_generation_prompt=True,
                            tokenize=True
                        )
                    except Exception as e:
                        print(f"Warning: Standard chat template failed with error: {e}")
                        print(f"Trying alternative template format without system role...")
                        # Fallback for models that don't support system role
                        try:
                            # Combine system prompt with user content
                            combined_prompt = f"{STUDENT_SYSTEM_PROMPT}\n\n{prompt_text}"
                            token_ids = student_tokenizer.apply_chat_template(
                                [
                                    {"role": "user", "content": combined_prompt}
                                ],
                                add_bos=True,
                                add_generation_prompt=True,
                                tokenize=True
                            )
                        except Exception as e2:
                            print(f"Warning: Alternative chat template also failed: {e2}")
                            print(f"Using basic tokenization...")
                            # Final fallback - just tokenize directly
                            combined_prompt = f"{STUDENT_SYSTEM_PROMPT}\n\n{prompt_text}"
                            token_ids = student_tokenizer.encode(combined_prompt)
                    
                    tokenized_no_hint_prompts.append(token_ids)

                # Run model inference on tokenized prompts
                outputs_no_hint = student_llm.generate(
                    prompt_token_ids=tokenized_no_hint_prompts,
                    sampling_params=student_params
                )

                # Organize results and evaluate
                results_no_hint = build_hf_results(
                    samples,
                    outputs_no_hint,
                    prompt_id_map_no_hint,
                    hints_list=None,
                    include_hint=False,
                    prompt_name=None  # No prompt name for no-hint results
                )
                results_no_hint = evaluate_predictions(
                    results_no_hint,
                    use_judge=args.use_judge,
                    evaluator=evaluator
                )
                
                # Save no-hint results
                base_name, ext = os.path.splitext(base_output_file)
                no_hint_file = f"{base_name}_{student_tag}_no_hint{ext}"
                with open(no_hint_file, "w") as f:
                    json.dump(results_no_hint, f, indent=2)
                print(f"[{student_tag}] Saved NO-HINT results to {no_hint_file}")
                
                # Mark this student model as having saved its no-hint results
                no_hint_saved[student_tag] = True
                
                # Clean up to free memory before with-hint inference
                del results_no_hint
                del outputs_no_hint
                del tokenized_no_hint_prompts
                gc.collect()
                # Don't delete student_llm and tokenizer - we'll use them for with-hint

            # --- WITH-HINT Inference ---
            # If student model was already loaded for no-hint, we'll reuse it
            # Otherwise we need to load it now
            if prompt_idx > 0 or not no_hint_saved[student_tag]:
                print(f"\n=== Student Model: {model_name} ===")
                student_llm = load_model(model_name)
                student_tokenizer = load_tokenizer(model_name)
                
                # Make sampling params from the student's config
                student_params = SamplingParams(
                    temperature=stu_temp,
                    top_p=stu_top_p,
                    max_tokens=stu_max_tokens
                )
            
            print(f"[{student_tag}] Generating answers (WITH HINT from {prompt_name})...")
            
            # Convert raw prompts to tokenized chat format
            tokenized_with_hint_prompts = []
            for prompt_text in with_hint_prompts:
                try:
                    # Try with system role first
                    token_ids = student_tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt_text}
                        ],
                        add_bos=True,
                        add_generation_prompt=True,
                        tokenize=True
                    )
                except Exception as e:
                    print(f"Warning: Standard chat template failed with error: {e}")
                    print(f"Trying alternative template format without system role...")
                    # Fallback for models that don't support system role
                    try:
                        # Combine system prompt with user content
                        combined_prompt = f"{STUDENT_SYSTEM_PROMPT}\n\n{prompt_text}"
                        token_ids = student_tokenizer.apply_chat_template(
                            [
                                {"role": "user", "content": combined_prompt}
                            ],
                            add_bos=True,
                            add_generation_prompt=True,
                            tokenize=True
                        )
                    except Exception as e2:
                        print(f"Warning: Alternative chat template also failed: {e2}")
                        print(f"Using basic tokenization...")
                        # Final fallback - just tokenize directly
                        combined_prompt = f"{STUDENT_SYSTEM_PROMPT}\n\n{prompt_text}"
                        token_ids = student_tokenizer.encode(combined_prompt)
                
                tokenized_with_hint_prompts.append(token_ids)

            # Run model inference on tokenized prompts
            outputs_with_hint = student_llm.generate(
                prompt_token_ids=tokenized_with_hint_prompts,
                sampling_params=student_params
            )

            # Organize results and evaluate
            results_with_hint = build_hf_results(
                samples,
                outputs_with_hint,
                prompt_id_map_with_hint,
                hints_list=hints,
                include_hint=True,
                prompt_name=prompt_name  # Include which teacher prompt was used
            )
            results_with_hint = evaluate_predictions(
                results_with_hint,
                use_judge=args.use_judge,
                evaluator=evaluator
            )

            # Save with-hint output for this prompt
            base_name, ext = os.path.splitext(base_output_file)
            with_hint_file = f"{base_name}_{student_tag}_{prompt_name}_with_hint{ext}"
            with open(with_hint_file, "w") as f:
                json.dump(results_with_hint, f, indent=2)
            print(f"[{student_tag}] Saved {prompt_name} WITH-HINT results to {with_hint_file}")

            # Free up memory for next student model
            del student_llm
            del student_tokenizer
            del results_with_hint
            del outputs_with_hint
            del tokenized_with_hint_prompts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Clear hints to free memory before next prompt
        del hints
        gc.collect()

    print("\nAll models and prompts completed. Exiting.")


if __name__ == "__main__":
    main()