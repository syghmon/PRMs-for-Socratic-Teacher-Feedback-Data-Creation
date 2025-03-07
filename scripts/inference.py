import json
import argparse
import os
import pathlib
import sys
import gc
import torch  # for torch.cuda.empty_cache(), if you're using CUDA
from transformers import AutoTokenizer
from requests.exceptions import ConnectionError
from vllm import LLM, SamplingParams

from utils.math_eval import MathEvaluator  # optional for LLM-based judge
from utils.evaluation_utils import evaluate_predictions  # single entry point for scoring

# Get the project root directory (parent of scripts)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()

# -------------------------------------------------------------------------
# System prompts for teacher and student (SmolLM2 chat style)
# -------------------------------------------------------------------------
TEACHER_SYSTEM_PROMPT = (
    "You are an expert math teacher. Provide a short, helpful hint to guide the user "
    "towards the solution. Do not reveal the final answer."
)

STUDENT_SYSTEM_PROMPT = (
    "You are a helpful and accurate math assistant. You explain your reasoning step by step "
    "and always give correct answers."
)


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
    return AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)


def build_hf_results(samples, outputs, prompt_map, hints_list=None, include_hint=False):
    """
    Build a huggingface-style results list:
    Each problem is one record with:
      - "problem"
      - "final_answer" (ground truth)
      - "responses" (list of model outputs, length = num_samples)
      - optionally "teacher_hint" if include_hint = True
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

        # Copy over any metadata you want to preserve
        if "difficulty_bin" in item:
            entry["difficulty_bin"] = item["difficulty_bin"]
        if "llama8b_solve_rate" in item:
            entry["llama8b_solve_rate"] = item["llama8b_solve_rate"]

        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Teacher–Student inference on math problems, with all config from a JSON file."
    )
    parser.add_argument("--input", "-i", type=str, default="data/samples.json",
                        help="Path to the JSON file of sampled problems.")
    parser.add_argument("--output", "-o", type=str, default="results/answers.json",
                        help="Base path for output files. Script appends model tags and '_no_hint'/'_with_hint'.")
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
    teacher_hint_prompt = cfg["teacher_hint_prompt"]
    teacher_temperature = cfg.get("teacher_temperature", 0.7)
    teacher_top_p = cfg.get("teacher_top_p", 0.9)
    teacher_max_tokens = cfg.get("teacher_max_tokens", 256)
    num_samples = cfg.get("num_samples", 1)

    student_configs = cfg["student_models"]  # list of dicts w/ model_name, temperature, top_p, etc.

    # -------------------------------------------------------------------------
    # 3) TEACHER Inference (Generate Hints) using Chat Format
    # -------------------------------------------------------------------------
    print("\n=== Teacher Model Inference ===")
    teacher_llm = load_model(teacher_model_name)
    teacher_tokenizer = load_tokenizer(teacher_model_name)

    # Build teacher prompts using chat format
    print(f"Preparing teacher prompt templates for {len(samples)} problems...")
    tokenized_teacher_prompts = []
    
    for item in samples:
        problem_text = item["problem"]
        # Combine teacher_hint_prompt with the problem
        user_content = f"{teacher_hint_prompt}\n\nProblem:\n{problem_text}"
        
        # Apply chat template to create tokenized prompt
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
        hint_text = output_obj.outputs[0].text.strip()
        hints.append(hint_text)

    # Free up teacher model memory
    del teacher_llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4) Prepare Student Prompts (No-hint and With-hint versions)
    # -------------------------------------------------------------------------
    # Create two sets of prompts:
    # 1. Problems without hints
    # 2. Problems with teacher-generated hints
    
    # a) NO-HINT prompts
    no_hint_prompts = []
    prompt_id_map_no_hint = {}

    for idx, item in enumerate(samples):
        problem_text = item["problem"]
        user_content = f"{problem_text}\n\nPlease reason step by step, and put your final answer in \\boxed{{}}."
        # For each problem, create num_samples identical prompts (for sampling diversity)
        for _ in range(num_samples):
            no_hint_prompts.append(user_content)
            # Track which problem index each prompt corresponds to
            prompt_id_map_no_hint[len(no_hint_prompts) - 1] = idx

    # b) WITH-HINT prompts
    with_hint_prompts = []
    prompt_id_map_with_hint = {}
    
    for idx, item in enumerate(samples):
        problem_text = item["problem"]
        hint_text = hints[idx]
        user_content = (
            f"{problem_text}\n\n"
            f"**HINT**:\n{hint_text}\n\n"
            "Please reason step by step, taking into account the hint, "
            "and put your final answer in \\boxed{{}}."
        )
        # For each problem, create num_samples identical prompts (for sampling diversity)
        for _ in range(num_samples):
            with_hint_prompts.append(user_content)
            # Track which problem index each prompt corresponds to
            prompt_id_map_with_hint[len(with_hint_prompts) - 1] = idx

    # Optionally, if user wants LLM-based judge, build the evaluator
    evaluator = None
    if args.use_judge:
        evaluator = MathEvaluator()

    # -------------------------------------------------------------------------
    # 5) For each Student Model: run inference with and without hints
    # -------------------------------------------------------------------------
    for stu_cfg in student_configs:
        model_name = stu_cfg["model_name"]
        stu_temp = stu_cfg.get("temperature", 0.6)
        stu_top_p = stu_cfg.get("top_p", 0.95)
        stu_max_tokens = stu_cfg.get("max_tokens", 1024)

        student_tag = safe_model_id(model_name)

        print(f"\n=== Student Model: {model_name} ===")
        student_llm = load_model(model_name)
        student_tokenizer = load_tokenizer(model_name)

        # Make sampling params from the student's config
        student_params = SamplingParams(
            temperature=stu_temp,
            top_p=stu_top_p,
            max_tokens=stu_max_tokens
        )

        # --- NO-HINT Inference ---
        print(f"[{student_tag}] Generating answers (NO HINT)...")
        # Convert raw prompts to tokenized chat format
        tokenized_no_hint_prompts = []
        for prompt_text in no_hint_prompts:
            token_ids = student_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt_text}
                ],
                add_bos=True,
                add_generation_prompt=True,
                tokenize=True
            )
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
            include_hint=False
        )
        results_no_hint = evaluate_predictions(
            results_no_hint,
            use_judge=args.use_judge,
            evaluator=evaluator
        )

        # --- WITH-HINT Inference ---
        print(f"[{student_tag}] Generating answers (WITH HINT)...")
        # Convert raw prompts to tokenized chat format
        tokenized_with_hint_prompts = []
        for prompt_text in with_hint_prompts:
            token_ids = student_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt_text}
                ],
                add_bos=True,
                add_generation_prompt=True,
                tokenize=True
            )
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
            include_hint=True
        )
        results_with_hint = evaluate_predictions(
            results_with_hint,
            use_judge=args.use_judge,
            evaluator=evaluator
        )

        # Save outputs
        base_name, ext = os.path.splitext(base_output_file)
        no_hint_file = f"{base_name}_{student_tag}_no_hint{ext}"
        with_hint_file = f"{base_name}_{student_tag}_with_hint{ext}"

        with open(no_hint_file, "w") as f:
            json.dump(results_no_hint, f, indent=2)
        print(f"[{student_tag}] Saved NO-HINT results to {no_hint_file}")

        with open(with_hint_file, "w") as f:
            json.dump(results_with_hint, f, indent=2)
        print(f"[{student_tag}] Saved WITH-HINT results to {with_hint_file}")

        # Free up memory for this student before moving on
        del student_llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll models completed. Exiting.")


if __name__ == "__main__":
    main()