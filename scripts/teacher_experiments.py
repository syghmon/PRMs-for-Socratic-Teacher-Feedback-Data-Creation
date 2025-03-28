#!/usr/bin/env python3
import json
import argparse
import os
import pathlib
import sys
import gc
import torch
import re
from transformers import AutoTokenizer
from requests.exceptions import ConnectionError
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

# -------------------------------------------------------------------------
# ENV
# -------------------------------------------------------------------------
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_HUG_TOKEN")
if hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["HF_TOKEN"] = hf_token
    print("Hugging Face token loaded from .env file")
else:
    print("Warning: HUGGINGFACE_TOKEN not found in .env file")

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()

# -------------------------------------------------------------------------
# TEACHER PARSING
# -------------------------------------------------------------------------
def parse_hint_blockquote(response_text: str) -> (str, bool):
    """
    For 'blockquote' parse_mode: capture lines after the first line that matches '> ...Hint...'
    until a line does not start with '>'.
    Returns (hint_text, parse_failed).
    """
    lines = response_text.splitlines()
    in_hint = False
    collected = []
    for line in lines:
        if not in_hint and re.search(r"^>\s*(.*hint.*)$", line, re.IGNORECASE):
            in_hint = True
            collected.append(re.sub(r"^>\s*", "", line))
        elif in_hint:
            if line.strip().startswith(">"):
                collected.append(re.sub(r"^>\s*", "", line))
            else:
                break
    if collected:
        return ("\n".join(collected).strip(), False)
    return ("Warning: No blockquote hint found.", True)

def parse_teacher_output(response_text: str, parse_mode: str):
    """
    Returns (hint, parse_failed).
    - 'none': no teacher inference
    - 'blockquote': parse blockquote
    """
    pm = parse_mode.lower()
    if pm == "none":
        return (None, False)
    elif pm == "blockquote":
        return parse_hint_blockquote(response_text)
    else:
        return (f"Unknown parse_mode '{parse_mode}'.", True)

# -------------------------------------------------------------------------
# STUDENT PARSING
# -------------------------------------------------------------------------
def parse_student_answer_answer_colon(response_text: str) -> (str, bool):
    match = re.search(r"Answer:\s*(.*)", response_text, re.IGNORECASE)
    if match:
        return (match.group(1).strip(), False)
    return ("No 'Answer:' found.", True)

def parse_student_answer_json(response_text: str) -> (str, bool):
    try:
        data = json.loads(response_text)
        fa = data.get("final_answer", None)
        if fa is not None:
            return (fa.strip(), False)
        else:
            return ("No 'final_answer' key in JSON.", True)
    except Exception as e:
        return (f"JSON parse error: {e}", True)

def parse_student_answer_unique_token(response_text: str) -> (str, bool):
    if "<unqd>" not in response_text:
        return ("No <unqd> token found.", True)
    chunks = response_text.split("<unqd>")
    last_chunk = chunks[-1].strip()
    if not last_chunk and len(chunks) > 1:
        last_chunk = chunks[-2].strip()
    if last_chunk:
        return (last_chunk, False)
    return ("Couldn't extract final answer from <unqd> tokens.", True)

def parse_student_answer_tagged(response_text: str) -> (str, bool):
    match = re.search(r"<final_answer>(.*?)</final_answer>", response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return (match.group(1).strip(), False)
    return ("No <final_answer>...</final_answer> found.", True)

def parse_student_answer_boxed(response_text: str) -> (str, bool):
    """
    Extracts answers formatted with LaTeX \boxed{} from the response.
    
    Args:
        response_text (str): The response text to parse.
        
    Returns:
        tuple: (extracted_answer, parse_failed_flag)
    """
    print("response_text: ", response_text)
    # Pattern to match \boxed{...} expressions
    match = re.search(r"\\boxed\{([^}]*)\}", response_text)
    if match:
        return (match.group(1).strip(), False)
    return ("No '\\boxed{}' expression found.", True)

def parse_student_answer(response_text: str, parse_mode: str) -> (str, bool):
    pm = parse_mode.lower()
    if pm == "answer_colon":
        return parse_student_answer_answer_colon(response_text)
    elif pm == "json":
        return parse_student_answer_json(response_text)
    elif pm == "unique_token":
        return parse_student_answer_unique_token(response_text)
    elif pm == "tagged_answer":
        return parse_student_answer_tagged(response_text)
    elif pm == "boxed":
        return parse_student_answer_boxed(response_text)
    else:
        return (response_text.strip(), True)  # fallback

# -------------------------------------------------------------------------
def load_model(model_name: str) -> LLM:
    print(f"Loading model: {model_name}")
    try:
        return LLM(model=model_name, trust_remote_code=True)
    except ConnectionError as e:
        print(f"[Error] Connection error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Could not load model '{model_name}': {e}")
        sys.exit(1)

def load_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

def safe_approach_id(text: str) -> str:
    return text.replace(" ", "_").replace(":", "_").replace("/", "_").replace("\\", "_")

# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Teacher–Student with extended in-context examples for each student approach.")
    parser.add_argument("--input", "-i", default="data/samples.json", help="Path to the JSON of problems.")
    parser.add_argument("--output", "-o", default="results/experiments_incontext.json", help="Base path for output.")
    parser.add_argument("--config", "-c", default="scripts/config_teacher_student_incontext.json", help="Path to config.")
    args = parser.parse_args()

    # 1) Load input
    input_file = args.input if os.path.isabs(args.input) else os.path.join(PROJECT_ROOT, args.input)
    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    base_output_file = args.output if os.path.isabs(args.output) else os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(os.path.dirname(base_output_file), exist_ok=True)

    config_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    if not os.path.exists(config_path):
        print(f"[Error] Config file does not exist: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as cf:
        cfg = json.load(cf)

    # Teacher config
    teacher_model_name = cfg["teacher_model"]
    teacher_temperature = cfg.get("teacher_temperature", 0.7)
    teacher_top_p = cfg.get("teacher_top_p", 0.9)
    teacher_max_tokens = cfg.get("teacher_max_tokens", 512)
    teacher_approaches = cfg["teacher_approaches"]

    # Student config
    student_model_name = cfg["student_model"]
    student_temperature = cfg.get("student_temperature", 0.7)
    student_top_p = cfg.get("student_top_p", 0.95)
    student_max_tokens = cfg.get("student_max_tokens", 1024)
    student_approaches = cfg["student_approaches"]

    # Subfolders
    teacher_folder = os.path.join(os.path.dirname(base_output_file), "teacher_results_incontext")
    student_folder = os.path.join(os.path.dirname(base_output_file), "student_results_incontext")
    os.makedirs(teacher_folder, exist_ok=True)
    os.makedirs(student_folder, exist_ok=True)

    # 2) Load teacher model
    print("\n=== Loading Teacher Model ===")
    teacher_llm = load_model(teacher_model_name)
    teacher_tokenizer = load_tokenizer(teacher_model_name)

    teacher_sampling_params = SamplingParams(
        temperature=teacher_temperature,
        top_p=teacher_top_p,
        max_tokens=teacher_max_tokens
    )

    teacher_outputs = {}
    teacher_parse_stats = {}

    # --- Teacher Inference ---
    for t_ap in teacher_approaches:
        t_name = t_ap["name"]
        t_prompt = t_ap["prompt"]
        t_parse_mode = t_ap.get("parse_mode", "none")

        print(f"\n=== Teacher Approach: {t_name} / parse_mode={t_parse_mode} ===")

        if t_parse_mode == "none":
            # no teacher inference, just placeholders
            approach_results = []
            for item in samples:
                approach_results.append({
                    "problem": item["problem"],
                    "ground_truth_answer": item.get("answer", ""),
                    "teacher_raw_output": "N/A",
                    "parsed_hint": None,
                    "parse_failed": False
                })
            teacher_outputs[t_name] = approach_results
            teacher_parse_stats[t_name] = (0, len(samples))
            print("No-hint approach. No teacher model call needed.")
            continue

        # Otherwise do normal inference
        tokenized_prompts = []
        for item in samples:
            problem_text = item["problem"]
            solution_text = item.get("answer", "")
            user_content = f"Problem:\n{problem_text}\n"
            if solution_text:
                user_content += f"\nSolution:\n{solution_text}\n"
            user_content += "\n" + t_prompt + "\n"

            token_ids = teacher_tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_content}
                ],
                add_bos=True,
                add_generation_prompt=True,
                tokenize=True
            )
            tokenized_prompts.append(token_ids)

        print(f"  Generating teacher outputs with approach '{t_name}' ...")
        outputs = teacher_llm.generate(
            prompt_token_ids=tokenized_prompts,
            sampling_params=teacher_sampling_params
        )

        approach_results = []
        fail_count = 0
        total = len(outputs)
        for i, out_obj in enumerate(outputs):
            raw_text = out_obj.outputs[0].text.strip()
            parsed_hint, parse_failed = parse_teacher_output(raw_text, t_parse_mode)
            if parse_failed:
                fail_count += 1
            approach_results.append({
                "problem": samples[i]["problem"],
                "ground_truth_answer": samples[i].get("answer", ""),
                "teacher_raw_output": raw_text,
                "parsed_hint": parsed_hint,
                "parse_failed": parse_failed
            })

        teacher_outputs[t_name] = approach_results
        teacher_parse_stats[t_name] = (fail_count, total)
        print(f"  parse-fail = {fail_count}/{total} for teacher approach '{t_name}'")

        # Save partial
        t_tag = safe_approach_id(t_name)
        partial_file = os.path.join(teacher_folder, f"{t_tag}_teacher.json")
        with open(partial_file, "w", encoding="utf-8") as pf:
            json.dump(approach_results, pf, indent=2)
        print(f"  Saved teacher partial results to {partial_file}")

    # Unload teacher
    print("\nUnloading teacher model.")
    del teacher_llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3) Load student model
    print(f"\n=== Loading Student Model: {student_model_name} ===")
    student_llm = load_model(student_model_name)
    student_tokenizer = load_tokenizer(student_model_name)

    student_sampling_params = SamplingParams(
        temperature=student_temperature,
        top_p=student_top_p,
        max_tokens=student_max_tokens
    )

    all_results = {}
    student_parse_stats = {}

    # 4) Student Inference
    for t_ap_name, t_data in teacher_outputs.items():
        print(f"\n--- Student testing with teacher approach: {t_ap_name} ---")
        approach_map = {}

        for s_ap in student_approaches:
            s_name = s_ap["name"]
            s_prompt = s_ap["system_prompt"]
            s_parse_mode = s_ap.get("parse_mode", "answer_colon")

            print(f"   Student approach: {s_name} (parse_mode={s_parse_mode})")

            tokenized_prompts = []
            index_map = {}
            for i, item in enumerate(t_data):
                problem_text = item["problem"]
                hint_text = item["parsed_hint"]
                if hint_text:
                    user_content = f"{problem_text}\n\nHere is a hint:\n{hint_text}\n\nPlease solve the problem."
                else:
                    user_content = f"{problem_text}\n\nNo hint provided. Please solve."

                token_ids = student_tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": s_prompt},
                        {"role": "user",   "content": user_content}
                    ],
                    add_bos=True,
                    add_generation_prompt=True,
                    tokenize=True
                )
                tokenized_prompts.append(token_ids)
                index_map[len(tokenized_prompts) - 1] = i

            s_outputs = student_llm.generate(
                prompt_token_ids=tokenized_prompts,
                sampling_params=student_sampling_params
            )

            final_entries = []
            parse_fails = 0
            total_count = len(s_outputs)

            for out_idx, gen_obj in enumerate(s_outputs):
                i_problem = index_map[out_idx]
                raw_text = gen_obj.outputs[0].text
                parsed_answer, parse_failed = parse_student_answer(raw_text, s_parse_mode)
                if parse_failed:
                    parse_fails += 1

                entry = {
                    "problem": t_data[i_problem]["problem"],
                    "ground_truth_answer": t_data[i_problem]["ground_truth_answer"],
                    "teacher_parsed_hint": t_data[i_problem]["parsed_hint"],
                    "student_raw_output": raw_text,
                    "parsed_answer": parsed_answer,
                    "parse_failed": parse_failed
                }
                final_entries.append(entry)

            approach_map[s_name] = final_entries
            student_parse_stats[(t_ap_name, s_name)] = (parse_fails, total_count)
            print(f"      parse-fail = {parse_fails}/{total_count}")

            # Save partial
            s_tag = safe_approach_id(s_name)
            t_tag = safe_approach_id(t_ap_name)
            partial_file = os.path.join(student_folder, f"{t_tag}__{s_tag}_student.json")
            with open(partial_file, "w", encoding="utf-8") as pf:
                json.dump(final_entries, pf, indent=2)

        all_results[t_ap_name] = approach_map

    # Unload student
    print("\nUnloading student model.")
    del student_llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5) Save combined results + summary
    combined_file = f"{os.path.splitext(base_output_file)[0]}_ALL.json"
    with open(combined_file, "w", encoding="utf-8") as cf:
        json.dump(all_results, cf, indent=2)
    print(f"Saved combined teacher–student results to {combined_file}")

    summary = {
        "teacher_parse_stats": {},
        "student_parse_stats": {}
    }
    for approach_name, (fail_count, total) in teacher_parse_stats.items():
        ratio = fail_count / total if total else 0
        summary["teacher_parse_stats"][approach_name] = {
            "fail_count": fail_count,
            "total": total,
            "fail_ratio": round(ratio, 3)
        }
    for (t_ap, s_ap), (fail_count, total) in student_parse_stats.items():
        ratio = fail_count / total if total else 0
        summary["student_parse_stats"][f"{t_ap} | {s_ap}"] = {
            "fail_count": fail_count,
            "total": total,
            "fail_ratio": round(ratio, 3)
        }

    summary_file = f"{os.path.splitext(base_output_file)[0]}_SUMMARY.json"
    with open(summary_file, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_file}")
    print("Done!")


if __name__ == "__main__":
    main()
