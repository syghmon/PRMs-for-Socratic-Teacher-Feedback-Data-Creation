import json
import argparse
import os
import pathlib
import sys
import gc

import torch  # for torch.cuda.empty_cache(), if you're using CUDA
from vllm import LLM, SamplingParams
from requests.exceptions import ConnectionError
from utils.math_eval import MathEvaluator  # optional for LLM-based judge
from utils.evaluation_utils import evaluate_predictions  # single entry point for scoring

# Get the project root directory (parent of scripts)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()


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
    Attempt to load a model with vLLM from Hugging Face.
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

    student_configs = cfg["student_models"]  # list of dicts, each with model_name, temperature, top_p, max_tokens, etc.

    # -------------------------------------------------------------------------
    # 3) TEACHER Inference (hints)
    # -------------------------------------------------------------------------
    print("\n=== Teacher Model Inference ===")
    teacher_llm = load_model(teacher_model_name)

    # Build teacher prompts (1 prompt per problem)
    teacher_prompts = []
    for item in samples:
        problem_text = item["problem"]
        prompt = f"{teacher_hint_prompt}\n\nProblem:\n{problem_text}\n"
        teacher_prompts.append(prompt)

    teacher_sampling_params = SamplingParams(
        temperature=teacher_temperature,
        top_p=teacher_top_p,
        max_tokens=teacher_max_tokens
    )

    print(f"Generating teacher hints for {len(samples)} problems...")
    teacher_outputs = teacher_llm.generate(teacher_prompts, teacher_sampling_params)

    # Map each problem to exactly one hint
    hints = []
    for idx, output_obj in enumerate(teacher_outputs):
        hint_text = output_obj.outputs[0].text.strip()
        hints.append(hint_text)

    # (Optional) free up teacher model
    # If you won't use the teacher model again, free it now.
    del teacher_llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4) Prepare Student Prompts (No-hint and With-hint)
    #    We'll just store them once; each model will re-run them with
    #    model-specific decoding parameters from the config.
    # -------------------------------------------------------------------------
    # a) No-hint
    no_hint_prompts = []
    prompt_id_map_no_hint = {}
    for idx, item in enumerate(samples):
        problem_text = item["problem"]
        prompt = (
            f"{problem_text}\n\n"
            "<think>\n"
            "Please reason step by step, and put your final answer in \\boxed{}."
        )
        for _ in range(num_samples):
            no_hint_prompts.append(prompt)
            prompt_id_map_no_hint[len(no_hint_prompts) - 1] = idx

    # b) With-hint
    with_hint_prompts = []
    prompt_id_map_with_hint = {}
    for idx, item in enumerate(samples):
        problem_text = item["problem"]
        hint_text = hints[idx]
        prompt = (
            f"{problem_text}\n\n"
            f"**HINT from teacher**:\n{hint_text}\n\n"
            "<think>\n"
            "Please reason step by step, taking into account the hint, and put your final answer in \\boxed{}."
        )
        for _ in range(num_samples):
            with_hint_prompts.append(prompt)
            prompt_id_map_with_hint[len(with_hint_prompts) - 1] = idx
    
    # Optionally, if user wants LLM-based judge, build the evaluator
    evaluator = None
    if args.use_judge:
        evaluator = MathEvaluator()
    # -------------------------------------------------------------------------
    # 5) For each Student Model: run no-hint + with-hint, then free memory
    #    We'll store results in "Hugging Face" format: one record per problem
    #    with a "responses" array of length = num_samples.
    # -------------------------------------------------------------------------
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

        for out_idx, output_obj in enumerate(outputs):
            problem_idx = prompt_map[out_idx]
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

    for stu_cfg in student_configs:
        model_name = stu_cfg["model_name"]
        stu_temp = stu_cfg.get("temperature", 0.6)
        stu_top_p = stu_cfg.get("top_p", 0.95)
        stu_max_tokens = stu_cfg.get("max_tokens", 1024)

        student_tag = safe_model_id(model_name)

        print(f"\n=== Student Model: {model_name} ===")
        student_llm = load_model(model_name)

        # Make sampling params from the student's config
        student_params = SamplingParams(
            temperature=stu_temp,
            top_p=stu_top_p,
            max_tokens=stu_max_tokens
        )

        # --- NO-HINT inference and evaluation
        print(f"[{student_tag}] Generating answers (NO HINT)...")
        outputs_no_hint = student_llm.generate(no_hint_prompts, student_params)
        results_no_hint = build_hf_results(
            samples, outputs_no_hint, prompt_id_map_no_hint,
            hints_list=None,
            include_hint=False
        )
        results_no_hint = evaluate_predictions(
            results_no_hint,
            use_judge=args.use_judge,
            evaluator=evaluator
        )

        # --- WITH-HINT inference and evaluation
        print(f"[{student_tag}] Generating answers (WITH HINT)...")
        outputs_with_hint = student_llm.generate(with_hint_prompts, student_params)
        results_with_hint = build_hf_results(
            samples, outputs_with_hint, prompt_id_map_with_hint,
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

        # Now free up memory from this student before moving on
        del student_llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll models completed. Exiting.")


if __name__ == "__main__":
    main()
