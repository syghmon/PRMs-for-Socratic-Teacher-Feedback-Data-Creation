#!/usr/bin/env python3
"""
analysis.py

Collects the JSON outputs from inference.py for student models (with and without hints),
plus the original samples.json, and generates:

1. Bin-based combined solve rates (showing improvement from hints for different models)
2. Interesting examples (problems where models showed significant improvement with hints)
3. Answer extraction success rates (checking if models produce boxed answers)

Usage:
  python analysis.py --samples data/joined_samples.json \
                     --results_dir results \
                     --output_dir analysis_outputs

Where:
  --samples/-s     : Path to original input problems (contains final_answer, difficulty_bin, llama8b_solve_rate).
  --results_dir/-r : Directory where the *_no_hint.json and *_with_hint.json are located.
  --output_dir/-o  : Directory to save the plots and interesting examples.
"""
import argparse
import os
import json
import sys
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
from typing import List, Optional, Dict, Set, Tuple
import matplotlib.colors as mc
import colorsys

##############################################################################
# Answer Extraction Functions
##############################################################################

def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Extracts the last boxed expression from a string.

    Args:
        string (str): The input string.

    Returns:
        Optional[str]: The last boxed expression or None.
    """
    # Check for null input
    if string is None:
        return None
    
    # Handle various boxed formats
    boxed_formats = [
        # Standard \boxed{...} format
        (r"\\boxed\s*\{([^}]*)\}", lambda m: f"\\boxed{{{m.group(1)}}}"),
        # \boxed ... format (no braces)
        (r"\\boxed\s+([^\s$]+)", lambda m: f"\\boxed{{{m.group(1)}}}"),
        # \fbox{...} format
        (r"\\fbox\s*\{([^}]*)\}", lambda m: f"\\fbox{{{m.group(1)}}}"),
        # $\boxed{...}$ format with math delimiters
        (r"\$\\boxed\s*\{([^}]*)\}\$", lambda m: f"\\boxed{{{m.group(1)}}}"),
        # Plain "boxed answer: ..." format
        (r"boxed\s+answer\s*:\s*([^\.]+)", lambda m: f"\\boxed{{{m.group(1)}}}"),
        # The answer is: ...
        (r"the answer is\s*:\s*([^\.]+)", lambda m: f"\\boxed{{{m.group(1)}}}"),
        # "Therefore, ... " format - more aggressive matching
        (r"therefore[,:]?\s+([^\.]+)", lambda m: f"\\boxed{{{m.group(1)}}}")
    ]
    
    # Try each format, prioritizing standard formats first
    for pattern, formatter in boxed_formats:
        matches = re.findall(pattern, string, re.IGNORECASE)
        if matches:
            # Return the last match formatted properly
            return formatter(re.search(pattern, string, re.IGNORECASE))
    
    # Legacy approach as fallback
    idx = string.rfind("\\boxed")
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

    Args:
        s (str): The input string.

    Returns:
        Optional[str]: String without boxed formatting or None.
    """
    if s is None:
        return None
        
    # Handle various formats
    if "\\boxed " in s:
        left = "\\boxed "
        return s[len(left):]
    elif "\\boxed{" in s:
        left = "\\boxed{"
        if s[-1] == "}":
            return s[len(left):-1]
        else:
            # Handle malformed boxed expressions
            end_idx = s.find("}", len(left))
            if end_idx != -1:
                return s[len(left):end_idx]
            return s[len(left):]
    elif "\\fbox{" in s:
        left = "\\fbox{"
        if s[-1] == "}":
            return s[len(left):-1]
        else:
            # Handle malformed fbox expressions
            end_idx = s.find("}", len(left))
            if end_idx != -1:
                return s[len(left):end_idx]
            return s[len(left):]
    elif "boxed answer:" in s.lower():
        # Extract from "boxed answer: X" format
        matched = re.search(r"boxed\s+answer\s*:\s*([^\.]+)", s, re.IGNORECASE)
        if matched:
            return matched.group(1).strip()
    elif "the answer is:" in s.lower():
        # Extract from "the answer is: X" format
        matched = re.search(r"the answer is\s*:\s*([^\.]+)", s, re.IGNORECASE)
        if matched:
            return matched.group(1).strip()
    elif "therefore" in s.lower():
        # Try to extract from "Therefore, X" pattern - more aggressive
        matched = re.search(r"therefore[,:]?\s+([^\.]+)", s, re.IGNORECASE)
        if matched:
            return matched.group(1).strip()
    
    return s

def has_formatted_answer(answer: str) -> bool:
    """
    Checks if the answer contains a formatted solution (boxed expression).

    Args:
        answer (str): The answer string.

    Returns:
        bool: True if formatted answer exists, False otherwise.
    """
    try:
        # Check for various boxed formats
        if "\\boxed" in answer or "\\fbox" in answer:
            boxed = last_boxed_only_string(answer)
            if boxed and remove_boxed(boxed):
                return True
        
        # Check for other answer formats
        lower_answer = answer.lower()
        if "boxed answer:" in lower_answer or "the answer is:" in lower_answer:
            return True
            
        # More aggressive detection for final answers
        if "therefore" in lower_answer and "." in answer:
            return True
            
        return False
    except Exception:
        return False

def extract_boxed_expressions(string: str) -> List[str]:
    """
    Extracts all \boxed{...} and \boxed ... expressions from the string.
    More lenient than the original version to catch various formats.
    """
    if string is None:
        return []
        
    boxed_expressions = []

    # Standard \boxed{...} pattern
    pattern_braces = r"\\boxed\s*\{([^}]*)\}"
    boxed_expressions += re.findall(pattern_braces, string)

    # \boxed ... pattern (no braces)
    pattern_space = r"\\boxed\s+([^\s\$]+)"
    boxed_expressions += re.findall(pattern_space, string)
    
    # \fbox{...} pattern
    pattern_fbox = r"\\fbox\s*\{([^}]*)\}"
    boxed_expressions += re.findall(pattern_fbox, string)
    
    # $\boxed{...}$ with math delimiters
    pattern_math_delim = r"\$\\boxed\s*\{([^}]*)\}\$"
    boxed_expressions += re.findall(pattern_math_delim, string)
    
    # "Boxed answer: X" pattern
    pattern_boxed_answer = r"boxed\s+answer\s*:\s*([^\.]+)"
    boxed_expressions += re.findall(pattern_boxed_answer, string, re.IGNORECASE)
    
    # "The answer is: X" pattern
    pattern_answer_is = r"the answer is\s*:\s*([^\.]+)"
    boxed_expressions += re.findall(pattern_answer_is, string, re.IGNORECASE)
    
    # "Therefore, X" pattern - more aggressive
    pattern_therefore = r"therefore[,:]?\s+([^\.]+)"
    boxed_expressions += re.findall(pattern_therefore, string, re.IGNORECASE)

    return ["\\boxed{" + expr.strip() + "}" for expr in boxed_expressions]

##############################################################################
# Utility Functions
##############################################################################

def load_json_data(filepath, exit_on_error=False):
    """Load JSON from a file, or return empty list if file doesn't exist."""
    if not os.path.exists(filepath):
        print(f"[Warning] File not found: {filepath}")
        if exit_on_error:
            sys.exit(1)
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"[Error] Failed to parse JSON from {filepath}: {e}")
            if exit_on_error:
                sys.exit(1)
            return []

def get_solve_rate(entry):
    """
    Given a result entry from the `_no_hint.json` or `_with_hint.json`,
    return the fraction of correct answers across all samples.
    entry["correctness"] is a list of bools for each sample (if the inference used multiple samples).
    If not found, returns 0.0.
    """
    correctness = entry.get("correctness", [])
    if not correctness:
        # DEBUG - Print entry keys if no correctness found
        print(f"    DEBUG - No correctness in entry with keys: {list(entry.keys())}")
        # Look for other possible fields like correct_answer or similar
        for key in entry.keys():
            if 'correct' in key.lower():
                print(f"    DEBUG - Found potential correctness field: {key} = {entry[key]}")
        return 0.0
    
    # Ensure correctness values are booleans
    try:
        # Convert any non-boolean values (like strings) to booleans if possible
        parsed_correctness = []
        for val in correctness:
            if isinstance(val, bool):
                parsed_correctness.append(val)
            elif isinstance(val, str):
                parsed_correctness.append(val.lower() == "true")
            elif isinstance(val, (int, float)):
                parsed_correctness.append(bool(val))
            else:
                parsed_correctness.append(False)
        
        if parsed_correctness:
            return sum(parsed_correctness) / len(parsed_correctness)
        return 0.0
    except Exception as e:
        print(f"[Warning] Error parsing correctness values: {e}")
        return 0.0

def compute_stats(values):
    """Calculate mean and standard error for a list of values."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std_error = (variance / len(values)) ** 0.5
    else:
        std_error = 0.0
    return mean, std_error

def custom_sort_bins(bins):
    """Sort bins in numerical order based on the lower bound percentage."""
    def extract_percentage(bin_label):
        try:
            # Extract the first number from strings like "0-10%"
            return float(bin_label.split('-')[0])
        except (ValueError, IndexError):
            return float('inf')  # Put non-standard bins at the end
    return sorted(bins, key=extract_percentage)

def set_plot_style():
    """Set consistent plot style across all visualizations."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except ValueError:
        # Fall back to a similar style if seaborn-v0_8-whitegrid is not available
        warnings.warn("seaborn-v0_8-whitegrid style not found, falling back to seaborn-whitegrid")
        try:
            plt.style.use('seaborn-whitegrid')
        except ValueError:
            warnings.warn("seaborn-whitegrid style not found, using default style")
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

def adjust_color(base_color, amount=0.1):
    """Adjust a color to create variation for different hint types."""
    try:
        c = mc.to_rgb(base_color)
        h, l, s = colorsys.rgb_to_hls(*c)
        return colorsys.hls_to_rgb(h, min(1, l * (1 + amount)), s)
    except:
        return base_color  # Return original if conversion fails

##############################################################################
# Main Analysis Class
##############################################################################

class Analysis:
    """
    Collects data from:
      - The original samples (with fields: problem, answer, difficulty_bin, llama8b_solve_rate, etc.)
      - The multiple JSON result files from inference (for each model, with/without hints).

    Produces:
      - Bin-based solve rates combined plot
      - Answer extraction success rates
      - Interesting examples
    """

    def __init__(self,
                 samples_file: str,
                 results_dir: str,
                 output_dir: str,
                 debug_extraction: bool = False):
        self.samples_file = samples_file
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.debug_extraction = debug_extraction
        
        # Discover model tags and hint types from filenames
        self.model_tags, self.hint_types = discover_models_and_hints(results_dir)
        
        # Hardcoded bin labels in increments of 10%
        self.bin_labels = [
            "0-10%", "10-20%", "20-30%", "30-40%", "40-50%", 
            "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"
        ]
        
        # Define debug variables
        self.debug_problems = list(range(5)) if debug_extraction else []
        
        # Data structure to hold per-problem info
        self.all_results = {}

    def run(self):
        """High-level workflow."""
        print(f"[Analysis] Creating output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Print detected models and hint types
        print(f"[Analysis] Processing {len(self.model_tags)} models: {', '.join(self.model_tags)}")
        print(f"[Analysis] Processing {len(self.hint_types)} hint types: {', '.join(self.hint_types)}")
        print(f"[Analysis] Using bin labels: {', '.join(self.bin_labels)}")
        
        self.load_samples()
        if not self.all_results:
            print("[Error] No samples loaded. Exiting.")
            return
            
        self.load_model_results()
        # Check if we have any model results
        has_model_data = False
        for i in self.all_results:
            if self.all_results[i]["models"]:
                has_model_data = True
                break
        
        if not has_model_data:
            print("[Error] No model results loaded. Exiting.")
            return
        
        # Print extraction rate summary for debugging
        if self.debug_extraction:
            self.print_extraction_summary()
            
        self.plot_bin_based_solve_rates()
        self.plot_bin_based_extraction_rates()
        self.save_interesting_examples()
        print("[Analysis] Done.")

    def load_samples(self):
        """Load original samples (problems, final_answer, difficulty_bin, llama8b_solve_rate, etc.)."""
        print(f"[Analysis] Loading samples from {self.samples_file}")
        samples = load_json_data(self.samples_file, exit_on_error=True)
        for i, samp in enumerate(samples):
            self.all_results[i] = {
                "problem": samp["problem"],
                "final_answer": samp.get("answer", ""),
                "difficulty_bin": samp.get("difficulty_bin", None),
                "llama8b_solve_rate": samp.get("llama8b_solve_rate", 0.0),
                "teacher_hints": {},  # Will store hints by type
                "models": {}
            }

    def load_model_results(self):
        """
        For each model tag, load no-hint data and separate hint data by hint type.
        """
        models_loaded = 0
        
        for tag in self.model_tags:
            print(f"[Analysis] Loading model results for {tag}")
            
            # Load no-hint data
            no_hint_file = os.path.join(self.results_dir, f"answers_{tag}_no_hint.json")
            no_hint_data = load_json_data(no_hint_file)
            
            # Check if no-hint file exists
            print(f"  - Found no-hint file: {os.path.exists(no_hint_file)}")
            
            # Initialize model data for all problems (for this tag)
            for i in range(len(self.all_results)):
                if tag not in self.all_results[i]["models"]:
                    self.all_results[i]["models"][tag] = {
                        "no_hint": 0.0,
                        "hint_types": {hint_type: 0.0 for hint_type in self.hint_types},
                        "has_answer_no_hint": 0.0,
                        "has_answer_hint_types": {hint_type: 0.0 for hint_type in self.hint_types}
                    }
            
            # Process no-hint data if available
            if no_hint_data:
                print(f"  - Loaded {len(no_hint_data)} entries from no-hint file")
                for i, entry in enumerate(no_hint_data):
                    if i >= len(self.all_results):
                        break
                        
                    # Get solve rate
                    sr = get_solve_rate(entry)
                    self.all_results[i]["models"][tag]["no_hint"] = sr
                    
                    # Use extraction_success flags if available, otherwise recompute
                    extraction_flags = entry.get("extraction_success", [])
                    
                    if extraction_flags:
                        # Use pre-computed extraction success flags
                        extraction_rate = sum(extraction_flags) / len(extraction_flags) if extraction_flags else 0.0
                    else:
                        # Fallback to computing extraction success here
                        has_answer_count = 0
                        responses = entry.get("responses", [])
                        
                        for response in responses:
                            if has_formatted_answer(response):
                                has_answer_count += 1
                        
                        responses_count = len(responses)
                        extraction_rate = has_answer_count / responses_count if responses_count > 0 else 0.0
                    
                    self.all_results[i]["models"][tag]["has_answer_no_hint"] = extraction_rate
            
            # Process each hint type separately
            for hint_type in self.hint_types:
                hint_file = os.path.join(self.results_dir, f"answers_{tag}_{hint_type}_with_hint.json")
                if os.path.exists(hint_file):
                    hint_data = load_json_data(hint_file)
                    if hint_data:
                        print(f"  - Loaded {len(hint_data)} entries for hint type: {hint_type}")
                        
                        # DEBUG: Print a few entries
                        if len(hint_data) > 0:
                            first_entry = hint_data[0]
                            correctness = first_entry.get("correctness", [])
                            print(f"    DEBUG - First entry correctness: {correctness}")
                            print(f"    DEBUG - Entry keys: {list(first_entry.keys())}")
                            
                            # DEBUG: Check if the first few entries have correctness
                            for i in range(min(3, len(hint_data))):
                                entry = hint_data[i]
                                correctness = entry.get("correctness", [])
                                sr = get_solve_rate(entry)
                                print(f"    DEBUG - Entry {i} correctness: {correctness}")
                                print(f"    DEBUG - Entry {i} solve rate: {sr}")
                                print(f"    DEBUG - Entry {i} problem: {entry.get('problem', '')[:50]}...")
                        
                        # Track processed problems to avoid duplicates
                        processed_problems = set()
                        
                        for entry in hint_data:
                            # Find the correct problem index
                            problem_id = entry.get("problem_idx", None)
                            
                            # If problem_idx field exists, use it
                            if problem_id is not None and 0 <= problem_id < len(self.all_results):
                                i = problem_id
                            else:
                                # Otherwise, try to find by problem text
                                problem_text = entry.get("problem", "")
                                found = False
                                for j, result in self.all_results.items():
                                    if result["problem"] == problem_text:
                                        i = j
                                        found = True
                                        break
                                if not found:
                                    continue  # Skip if no matching problem found
                            
                            # Skip if we've already processed this problem for this hint type
                            if i in processed_problems:
                                continue
                            processed_problems.add(i)
                            
                            # Get solve rate for this hint type
                            sr = get_solve_rate(entry)
                            self.all_results[i]["models"][tag]["hint_types"][hint_type] = sr
                            
                            # Use extraction_success flags if available, otherwise recompute
                            extraction_flags = entry.get("extraction_success", [])
                            
                            if extraction_flags:
                                # Use pre-computed extraction success flags
                                extraction_rate = sum(extraction_flags) / len(extraction_flags) if extraction_flags else 0.0
                            else:
                                # Fallback to computing extraction success here
                                has_answer_count = 0
                                responses = entry.get("responses", [])
                                
                                for response in responses:
                                    if has_formatted_answer(response):
                                        has_answer_count += 1
                                
                                responses_count = len(responses)
                                extraction_rate = has_answer_count / responses_count if responses_count > 0 else 0.0
                            
                            self.all_results[i]["models"][tag]["has_answer_hint_types"][hint_type] = extraction_rate
                            
                            # Store the teacher hint text
                            if "teacher_hint" in entry and entry["teacher_hint"]:
                                self.all_results[i]["teacher_hints"][hint_type] = entry["teacher_hint"]
            
            # Skip this model if no data was found
            if not no_hint_data and not any(
                os.path.exists(os.path.join(self.results_dir, f"answers_{tag}_{hint_type}_with_hint.json"))
                for hint_type in self.hint_types
            ):
                print(f"[Warning] Skipping model {tag} as no data files were found")
                continue
            
            models_loaded += 1
        
        print(f"[Analysis] Loaded data for {models_loaded} models")

    def plot_bin_based_solve_rates(self):
        """
        Group problems by difficulty_bin, compute average solve rates for each group and hint type,
        and create a combined plot showing both no-hint and all hint types.
        """
        print("[Analysis] Plotting bin-based solve rates.")
        
        # Create a map: bin_label -> list of dict
        #   each dict has: {"llama8b": x, "ModelX_no": x, "ModelX_hint_type1": x, "ModelX_hint_type2": x, ...}
        bin_map = defaultdict(list)
        # Track counts per bin for annotation
        bin_counts = defaultdict(int)
        
        for i in range(len(self.all_results)):
            b = self.all_results[i]["difficulty_bin"]
            if b is None:
                continue
            bin_counts[b] += 1
            item = {
                "llama8b": self.all_results[i]["llama8b_solve_rate"]
            }
            for tag in self.model_tags:
                if tag in self.all_results[i]["models"]:
                    # Add no-hint solve rate
                    item[f"{tag}_no"] = self.all_results[i]["models"][tag]["no_hint"]
                    
                    # Add hint-specific solve rates
                    for hint_type in self.hint_types:
                        if hint_type in self.all_results[i]["models"][tag]["hint_types"]:
                            item[f"{tag}_{hint_type}"] = self.all_results[i]["models"][tag]["hint_types"][hint_type]
            
            bin_map[b].append(item)

        if not bin_map:
            print("[Warning] No binned data available for plotting. Skipping bin-based solve rates plot.")
            return
            
        # Prepare data with means and std errors
        bin_summary = {}
        for b, records in bin_map.items():
            if len(records) == 0:
                continue
            
            # LLaMA8B stats
            llama_vals = [r["llama8b"] for r in records]
            llama_mean, llama_stderr = compute_stats(llama_vals)
            
            result = {
                "llama8b_mean": llama_mean,
                "llama8b_stderr": llama_stderr,
                "count": len(records)
            }
            
            for tag in self.model_tags:
                # Check if this model has data for this bin
                if not all(f"{tag}_no" in r for r in records):
                    continue
                    
                # No hint stats
                no_vals = [r[f"{tag}_no"] for r in records if f"{tag}_no" in r]
                no_mean, no_stderr = compute_stats(no_vals)
                result[f"{tag}_no_mean"] = no_mean
                result[f"{tag}_no_stderr"] = no_stderr
                
                # Stats for each hint type
                for hint_type in self.hint_types:
                    hint_key = f"{tag}_{hint_type}"
                    hint_vals = [r[hint_key] for r in records if hint_key in r]
                    if hint_vals:
                        hint_mean, hint_stderr = compute_stats(hint_vals)
                        result[f"{tag}_{hint_type}_mean"] = hint_mean
                        result[f"{tag}_{hint_type}_stderr"] = hint_stderr
            
            bin_summary[b] = result

        # Sort bins properly
        all_bins = list(bin_summary.keys())
        if not all_bins:
            print("[Warning] No bin data available for plotting. Skipping bin-based solve rates plot.")
            return
            
        # Use bin_labels if provided, otherwise sort the found bins
        if self.bin_labels and all(b in bin_summary for b in self.bin_labels):
            bins_in_order = self.bin_labels
        else:
            try:
                bins_in_order = custom_sort_bins(all_bins)
            except Exception as e:
                print(f"[Warning] Error sorting bins: {e}. Using original order.")
                bins_in_order = all_bins
        
        # Print bin statistics for debug
        print("\n[Analysis] Bin Statistics:")
        print(f"{'Bin':<10} {'Count':>6} {'LLaMA8B':>10}", end="")
        for tag in self.model_tags:
            print(f" {tag+' (no)':>12}", end="")
            for hint_type in self.hint_types:
                print(f" {tag+' ('+hint_type+')':>15}", end="")
        print()
        print("-" * (10 + 6 + 10 + sum(12 + sum(15 for _ in self.hint_types) for _ in self.model_tags)))
        
        for b in bins_in_order:
            print(f"{b:<10} {bin_summary[b]['count']:>6} {bin_summary[b]['llama8b_mean']:>10.2f}", end="")
            for tag in self.model_tags:
                no_key = f"{tag}_no_mean"
                if no_key in bin_summary[b]:
                    print(f" {bin_summary[b][no_key]:>12.2f}", end="")
                else:
                    print(f" {'N/A':>12}", end="")
                
                for hint_type in self.hint_types:
                    hint_key = f"{tag}_{hint_type}_mean"
                    if hint_key in bin_summary[b]:
                        print(f" {bin_summary[b][hint_key]:>15.2f}", end="")
                    else:
                        print(f" {'N/A':>15}", end="")
            print()
        
        # Create a visualization showing no-hint and all hint types
        self.plot_bin_based_solve_rates_combined(bin_summary, bins_in_order)

    def plot_bin_based_solve_rates_combined(self, bin_summary, bins_in_order):
        """
        Create a combined visualization showing both no-hint and all hint types
        in a clearer format, focusing on the comparison between hint types.
        """
        try:
            set_plot_style()
            
            # Create line plot for all models, showing no-hint and each hint type
            fig, ax = plt.subplots(figsize=(12, 7))
            x_vals = range(len(bins_in_order))
            
            # Plot LLaMA8B baseline as a grey line
            llama_means = [bin_summary[b]["llama8b_mean"] for b in bins_in_order]
            ax.plot(x_vals, llama_means, 'o--', color='grey', linewidth=2, 
                    label="LLaMA8B", alpha=0.7, markersize=8)
            
            # Plot each model's no-hint and all hint types
            # Use a different color for each model and different markers/line styles for hint types
            model_colors = plt.cm.tab10.colors
            hint_markers = ['o', 's', '^', 'D', 'v']  # Different marker for each hint type
            hint_linestyles = ['-', '--', '-.', ':']  # Different line style for each hint type
            
            for i, tag in enumerate(self.model_tags):
                base_color = model_colors[i % len(model_colors)]
                
                # Check if we have no-hint data for this model
                if not all(f"{tag}_no_mean" in bin_summary[b] for b in bins_in_order):
                    print(f"[Warning] Skipping model {tag} in plot due to missing no-hint data")
                    continue
                
                # No hint (solid line)
                no_means = [bin_summary[b][f"{tag}_no_mean"] for b in bins_in_order]
                ax.plot(x_vals, no_means, marker=hint_markers[0], linestyle=hint_linestyles[0],
                        label=f"{tag} (no hint)", color=base_color, linewidth=2, markersize=8)
                
                # Each hint type (different markers and line styles)
                for j, hint_type in enumerate(self.hint_types):
                    # Check if we have data for this hint type
                    hint_key = f"{tag}_{hint_type}_mean"
                    if not all(hint_key in bin_summary[b] for b in bins_in_order):
                        print(f"[Warning] Skipping hint type {hint_type} for model {tag} due to missing data")
                        continue
                    
                    # Get hint means for this type
                    hint_means = [bin_summary[b][hint_key] for b in bins_in_order]
                    
                    # Create slightly different color for each hint type
                    hint_color = adjust_color(base_color, (j+1)*0.15)
                    
                    # Plot this hint type
                    marker_idx = (j+1) % len(hint_markers)
                    style_idx = (j+1) % len(hint_linestyles)
                    ax.plot(x_vals, hint_means, 
                            marker=hint_markers[marker_idx], 
                            linestyle=hint_linestyles[style_idx],
                            label=f"{tag} ({hint_type})", 
                            color=hint_color, linewidth=2, markersize=8)
                    
                    # Fill the area between no-hint and this hint type (light shading)
                    ax.fill_between(x_vals, no_means, hint_means, color=hint_color, alpha=0.1)
            
            # Improve plot appearance
            ax.set_xticks(list(x_vals))
            ax.set_xticklabels(bins_in_order, rotation=45, ha="right")
            ax.set_ylabel("Average Solve Rate")
            ax.set_ylim([0, 1])
            ax.set_xlabel("Difficulty Bin (LLaMA8B solve rate)")
            ax.set_title("Model Performance Comparison With Different Hint Types", fontweight='bold')
            
            # Add sample counts as annotations
            for i, b in enumerate(bins_in_order):
                count = bin_summary[b]["count"]
                ax.annotate(f"n={count}", xy=(i, 0.05), xytext=(0, 0), 
                           textcoords="offset points", ha='center', va='bottom',
                           color='black', fontweight='bold', fontsize=9)
            
            # Add grid and reference line
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
            
            # Better legend - place it outside the plot to avoid overcrowding
            ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5),
                     frameon=True, fancybox=True)
            
            # Adjust layout to make room for the legend
            plt.tight_layout()
            plt.subplots_adjust(right=0.75)  # Make more room for the legend
            
            # Save the combined plot
            out_path = os.path.join(self.output_dir, "bin_based_solve_rates_combined.png")
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  -> Saved combined bin-based solve rate plot to {out_path}")
        except Exception as e:
            print(f"[Error] Failed to generate bin-based solve rates plot: {e}")
            import traceback
            traceback.print_exc()

    def plot_answer_extraction_stats(self):
        """
        Create visualization showing answer extraction success rates for each model
        (i.e., whether the model produced a properly formatted answer in a boxed environment).
        """
        print("[Analysis] Plotting answer extraction success rates.")
        
        try:
            # Collect answer extraction stats for each model
            model_data = {}
            for tag in self.model_tags:
                no_hint_rates = []
                with_hint_rates = []
                
                for i in range(len(self.all_results)):
                    if tag in self.all_results[i]["models"]:
                        model_info = self.all_results[i]["models"][tag]
                        if "has_answer_no_hint" in model_info:
                            no_hint_rates.append(model_info["has_answer_no_hint"])
                        if "has_answer_with_hint" in model_info:
                            with_hint_rates.append(model_info["has_answer_with_hint"])
                
                if no_hint_rates and with_hint_rates:
                    no_hint_mean, no_hint_stderr = compute_stats(no_hint_rates)
                    with_hint_mean, with_hint_stderr = compute_stats(with_hint_rates)
                    
                    model_data[tag] = {
                        "no_hint_mean": no_hint_mean,
                        "no_hint_stderr": no_hint_stderr,
                        "with_hint_mean": with_hint_mean,
                        "with_hint_stderr": with_hint_stderr,
                        "extraction_diff": with_hint_mean - no_hint_mean
                    }
            
            if not model_data:
                print("[Warning] No answer extraction data available. Skipping answer extraction plot.")
                return
            
            # Print summary statistics
            print("\n[Analysis] Answer Extraction Rate Statistics:")
            print(f"{'Model':<15} {'No Hint':>10} {'With Hint':>10} {'Diff':>10}")
            print("-" * 50)
            for tag, stats in model_data.items():
                print(f"{tag:<15} {stats['no_hint_mean']:>10.2%} {stats['with_hint_mean']:>10.2%} {stats['extraction_diff']:>+10.2%}")
            
            # Create bar plot comparing extraction rates
            set_plot_style()
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Prepare data
            tags = list(model_data.keys())
            x = range(len(tags))
            width = 0.35
            
            # No hint bars
            no_hint_means = [model_data[tag]["no_hint_mean"] for tag in tags]
            no_hint_stderrs = [model_data[tag]["no_hint_stderr"] for tag in tags]
            
            # With hint bars
            with_hint_means = [model_data[tag]["with_hint_mean"] for tag in tags]
            with_hint_stderrs = [model_data[tag]["with_hint_stderr"] for tag in tags]
            
            # Plot bars
            rects1 = ax.bar([i - width/2 for i in x], no_hint_means, width, label='No Hint',
                           color='lightblue', yerr=no_hint_stderrs, capsize=5)
            rects2 = ax.bar([i + width/2 for i in x], with_hint_means, width, label='With Hint',
                           color='lightgreen', yerr=with_hint_stderrs, capsize=5)
            
            # Add value labels on top of bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.1%}',
                               xy=(rect.get_x() + rect.get_width()/2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
            
            autolabel(rects1)
            autolabel(rects2)
            
            # Add the improvement percentage as text between bars
            for i, tag in enumerate(tags):
                diff = model_data[tag]["extraction_diff"]
                if abs(diff) > 0.01:  # Only show differences greater than 1%
                    color = 'green' if diff > 0 else 'red'
                    ax.annotate(f'{diff:+.1%}',
                               xy=(i, max(no_hint_means[i], with_hint_means[i]) + 0.05),
                               ha='center', fontweight='bold', color=color)
            
            # Customize the plot
            ax.set_ylabel('Answer Extraction Success Rate')
            ax.set_title('Answer Extraction Success Rate by Model (With vs. Without Hints)', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(tags)
            ax.legend()
            ax.set_ylim(0, 1.05)  # Set y-axis to include space for labels
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            
            fig.tight_layout()
            
            # Save the plot
            out_path = os.path.join(self.output_dir, "answer_extraction_rates.png")
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  -> Saved answer extraction rates plot to {out_path}")
            
            # Create a JSON summary file with the extraction statistics
            summary_data = {
                "overall_stats": {
                    tag: {
                        "no_hint_extraction_rate": stats["no_hint_mean"],
                        "with_hint_extraction_rate": stats["with_hint_mean"],
                        "extraction_improvement": stats["extraction_diff"]
                    } for tag, stats in model_data.items()
                }
            }
            
            summary_path = os.path.join(self.output_dir, "answer_extraction_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)
            print(f"  -> Saved answer extraction summary to {summary_path}")
            
        except Exception as e:
            print(f"[Error] Failed to generate answer extraction stats plot: {e}")
            import traceback
            traceback.print_exc()

    def plot_bin_based_extraction_rates(self):
        """
        Group problems by difficulty_bin, compute average answer extraction success rates for each group,
        and create a combined plot showing both no-hint and all hint types across bins.
        """
        print("[Analysis] Plotting bin-based answer extraction success rates.")
        
        # Create a map: bin_label -> list of dict
        #   each dict has: {"ModelX_no_extract": x, "ModelX_hint_type1_extract": x, "ModelX_hint_type2_extract": x, ...}
        bin_map = defaultdict(list)
        # Also track counts per bin for annotation
        bin_counts = defaultdict(int)
        
        for i in range(len(self.all_results)):
            b = self.all_results[i]["difficulty_bin"]
            if b is None:
                continue
            bin_counts[b] += 1
            item = {}
            for tag in self.model_tags:
                if tag in self.all_results[i]["models"]:
                    # Add no-hint extraction rate
                    item[f"{tag}_no_extract"] = self.all_results[i]["models"][tag]["has_answer_no_hint"]
                    
                    # Add extraction rates for each hint type
                    for hint_type in self.hint_types:
                        if hint_type in self.all_results[i]["models"][tag]["has_answer_hint_types"]:
                            item[f"{tag}_{hint_type}_extract"] = self.all_results[i]["models"][tag]["has_answer_hint_types"][hint_type]
        
            bin_map[b].append(item)

        if not bin_map:
            print("[Warning] No binned data available for extraction rate plotting. Skipping bin-based extraction rates plot.")
            return
            
        # Prepare data with means and std errors
        bin_summary = {}
        for b, records in bin_map.items():
            if len(records) == 0:
                continue
            
            result = {
                "count": len(records)
            }
            
            for tag in self.model_tags:
                # Check if this model has data for this bin
                if not all(f"{tag}_no_extract" in r for r in records):
                    continue
                    
                # No hint stats
                no_vals = [r[f"{tag}_no_extract"] for r in records if f"{tag}_no_extract" in r]
                no_mean, no_stderr = compute_stats(no_vals)
                result[f"{tag}_no_extract_mean"] = no_mean
                result[f"{tag}_no_extract_stderr"] = no_stderr
                
                # Stats for each hint type
                for hint_type in self.hint_types:
                    extract_key = f"{tag}_{hint_type}_extract"
                    hint_vals = [r[extract_key] for r in records if extract_key in r]
                    if hint_vals:
                        hint_mean, hint_stderr = compute_stats(hint_vals)
                        result[f"{tag}_{hint_type}_extract_mean"] = hint_mean
                        result[f"{tag}_{hint_type}_extract_stderr"] = hint_stderr
            
            bin_summary[b] = result

        # Sort bins properly
        all_bins = list(bin_summary.keys())
        if not all_bins:
            print("[Warning] No bin data available for extraction rate plotting. Skipping bin-based extraction rates plot.")
            return
            
        # Use bin_labels if provided, otherwise sort the found bins
        if self.bin_labels and all(b in bin_summary for b in self.bin_labels):
            bins_in_order = self.bin_labels
        else:
            try:
                bins_in_order = custom_sort_bins(all_bins)
            except Exception as e:
                print(f"[Warning] Error sorting bins: {e}. Using original order.")
                bins_in_order = all_bins
        
        # Print bin statistics for debug
        print("\n[Analysis] Bin-Based Answer Extraction Rate Statistics:")
        print(f"{'Bin':<10} {'Count':>6}", end="")
        for tag in self.model_tags:
            print(f" {tag+' (no)':>12}", end="")
            for hint_type in self.hint_types:
                print(f" {tag+' ('+hint_type+')':>15}", end="")
        print()
        print("-" * (10 + 6 + sum(12 + sum(15 for _ in self.hint_types) for _ in self.model_tags)))
        
        for b in bins_in_order:
            print(f"{b:<10} {bin_summary[b]['count']:>6}", end="")
            for tag in self.model_tags:
                no_key = f"{tag}_no_extract_mean"
                if no_key in bin_summary[b]:
                    print(f" {bin_summary[b][no_key]:>12.2%}", end="")
                else:
                    print(f" {'N/A':>12}", end="")
            
                for hint_type in self.hint_types:
                    hint_key = f"{tag}_{hint_type}_extract_mean"
                    if hint_key in bin_summary[b]:
                        print(f" {bin_summary[b][hint_key]:>15.2%}", end="")
                    else:
                        print(f" {'N/A':>15}", end="")
            print()
        
        # Create a visualization showing both no-hint and all hint types
        try:
            set_plot_style()
            
            # Create line plot
            fig, ax = plt.subplots(figsize=(12, 7))
            x_vals = range(len(bins_in_order))
            
            # Plot each model with different hint types
            colors = plt.cm.tab10.colors
            markers = ['o', 's', '^', 'D', 'v']  # Different marker for each hint type
            linestyles = ['-', '--', '-.', ':']  # Different line style for each hint type
            
            for i, tag in enumerate(self.model_tags):
                base_color = colors[i % len(colors)]
                
                # Check if we have no-hint data for this model
                no_key = f"{tag}_no_extract_mean"
                if not all(no_key in bin_summary[b] for b in bins_in_order):
                    print(f"[Warning] Skipping model {tag} in extraction rate plot due to missing data")
                    continue
                
                # No hint (solid line)
                no_means = [bin_summary[b][no_key] for b in bins_in_order]
                ax.plot(x_vals, no_means, marker=markers[0], linestyle=linestyles[0],
                        label=f"{tag} (no hint)", color=base_color, linewidth=2, markersize=8)
                
                # Each hint type (different markers and line styles)
                for j, hint_type in enumerate(self.hint_types):
                    # Check if we have data for this hint type
                    hint_key = f"{tag}_{hint_type}_extract_mean"
                    if not all(hint_key in bin_summary[b] for b in bins_in_order):
                        print(f"[Warning] Skipping hint type {hint_type} for model {tag} in extraction plot")
                        continue
                    
                    # Get hint means for this type
                    hint_means = [bin_summary[b][hint_key] for b in bins_in_order]
                    
                    # Create slightly different color for each hint type
                    hint_color = adjust_color(base_color, (j+1)*0.15)
                    
                    # Plot this hint type
                    marker_idx = (j+1) % len(markers)
                    style_idx = (j+1) % len(linestyles)
                    ax.plot(x_vals, hint_means, 
                            marker=markers[marker_idx], 
                            linestyle=linestyles[style_idx],
                            label=f"{tag} ({hint_type})", 
                            color=hint_color, linewidth=2, markersize=8)
                    
                    # Fill the area between no-hint and this hint type (light shading)
                    ax.fill_between(x_vals, no_means, hint_means, color=hint_color, alpha=0.1)
            
            # Improve plot appearance
            ax.set_xticks(list(x_vals))
            ax.set_xticklabels(bins_in_order, rotation=45, ha="right")
            ax.set_ylabel('Answer Extraction Success Rate')
            ax.set_ylim([0, 1])
            ax.set_xlabel("Difficulty Bin (LLaMA8B solve rate)")
            ax.set_title("Answer Extraction Success Rate by Difficulty Bin", fontweight='bold')
            
            # Add sample counts as annotations
            for i, b in enumerate(bins_in_order):
                count = bin_summary[b]["count"]
                ax.annotate(f"n={count}", xy=(i, 0.05), xytext=(0, 0), 
                           textcoords="offset points", ha='center', va='bottom',
                           color='black', fontweight='bold', fontsize=9)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Better legend - place it outside the plot
            ax.legend(fontsize='small', loc='center left', 
                     bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True)
            
            # Add explanation text
            ax.text(0.5, -0.15, 
                   "Answer Extraction Success: Percentage of responses with proper \\boxed{} formatted answers",
                   transform=ax.transAxes, ha='center', fontsize=9, style='italic')
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(right=0.75)  # Make room for the legend
            
            # Save the combined plot
            out_path = os.path.join(self.output_dir, "bin_based_extraction_rates.png")
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  -> Saved bin-based answer extraction rate plot to {out_path}")
        except Exception as e:
            print(f"[Error] Failed to generate bin-based extraction rates plot: {e}")
            import traceback
            traceback.print_exc()

    def save_interesting_examples(self):
        """
        Find problems where at least one model had a large improvement with hint
        or some other interesting condition:
          - e.g. (no_hint==0) -> (with_hint==1)
          - or improvement > 0.5
          - or performance got significantly worse with a hint (improvement < -0.2)
        Save them to a JSON or CSV in output_dir for manual inspection.
        """
        print("[Analysis] Saving interesting examples.")
        interesting = []

        for i, item in self.all_results.items():
            prob_text = item["problem"]
            teacher_hints = item["teacher_hints"]
            for tag in self.model_tags:
                if tag not in item["models"]:
                    continue
                    
                no_score = item["models"][tag]["no_hint"]
                
                # Find the best score across all hint types
                hint_scores = item["models"][tag]["hint_types"]
                if not hint_scores:
                    continue
                    
                # Fix: incorrect syntax and logic in accessing hint scores
                # Original line had a syntax error with mismatched quotes and 
                # was trying to use 'tag' as a hint type
                with_score = max(hint_scores.values())  # Use the best hint score
                best_hint_type = max(hint_scores.items(), key=lambda x: x[1])[0] if hint_scores else None
                
                improvement = with_score - no_score
                
                # Also get answer extraction success rates
                has_answer_no_hint = item["models"][tag]["has_answer_no_hint"]
                has_answer_with_hint = item["models"][tag]["has_answer_hint_types"].get(best_hint_type, 0.0)
                extraction_improvement = has_answer_with_hint - has_answer_no_hint
                
                # Condition: big jump or from 0 to near 1
                if improvement >= 0.5 or (no_score < 0.1 and with_score > 0.8) or improvement <= -0.2:
                    example_type = "improvement" if improvement >= 0 else "regression"
                    example = {
                        "problem_index": i,
                        "problem": prob_text,
                        "teacher_hints": teacher_hints,
                        "model": tag,
                        "best_hint_type": best_hint_type,
                        "score_no_hint": no_score,
                        "score_with_hint": with_score,
                        "improvement": improvement,
                        "has_answer_no_hint": has_answer_no_hint,
                        "has_answer_with_hint": has_answer_with_hint,
                        "extraction_improvement": extraction_improvement,
                        "example_type": example_type,
                        "difficulty_bin": item["difficulty_bin"],
                        "llama8b_solve_rate": item["llama8b_solve_rate"],
                    }
                    interesting.append(example)

        # Sort descending by absolute improvement value to show both large improvements and large regressions at the top
        interesting_sorted = sorted(interesting, key=lambda x: abs(x["improvement"]), reverse=True)

        out_path = os.path.join(self.output_dir, "interesting_examples.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(interesting_sorted, f, indent=2)
            
            # Count improvements and regressions
            improvements = sum(1 for ex in interesting_sorted if ex["improvement"] >= 0)
            regressions = sum(1 for ex in interesting_sorted if ex["improvement"] < 0)
            
            print(f"  -> Saved {len(interesting_sorted)} interesting examples to {out_path}")
            print(f"     ({improvements} improvements, {regressions} regressions)")
            
            # Also save separate files for improvements and regressions
            if improvements > 0:
                improvements_path = os.path.join(self.output_dir, "improvements.json")
                with open(improvements_path, "w", encoding="utf-8") as f:
                    json.dump([ex for ex in interesting_sorted if ex["improvement"] >= 0], f, indent=2)
                print(f"  -> Saved {improvements} improvement examples to {improvements_path}")
                
            if regressions > 0:
                regressions_path = os.path.join(self.output_dir, "regressions.json")
                with open(regressions_path, "w", encoding="utf-8") as f:
                    json.dump([ex for ex in interesting_sorted if ex["improvement"] < 0], f, indent=2)
                print(f"  -> Saved {regressions} regression examples to {regressions_path}")
                
        except Exception as e:
            print(f"[Error] Failed to save interesting examples: {e}")

    def print_extraction_summary(self):
        """Print a summary of answer extraction success rates for debugging."""
        print("\n[Analysis] Answer Extraction Rate Summary:")
        print(f"{'Model':<20} {'No Hint':>15} {'With Hint (Best)':>20} {'Diff':>10}")
        print("-" * 70)
        
        for tag in self.model_tags:
            no_hint_total = 0
            no_hint_success = 0
            
            # For each hint type, track totals separately
            hint_totals = {hint_type: 0 for hint_type in self.hint_types}
            hint_success = {hint_type: 0 for hint_type in self.hint_types}
            
            for i in self.all_results:
                if tag in self.all_results[i]["models"]:
                    model_info = self.all_results[i]["models"][tag]
                    
                    # No hint data
                    if "has_answer_no_hint" in model_info:
                        no_hint_total += 1
                        no_hint_success += model_info["has_answer_no_hint"]
                    
                    # With hint data for each hint type
                    for hint_type in self.hint_types:
                        if hint_type in model_info["has_answer_hint_types"]:
                            hint_totals[hint_type] += 1
                            hint_success[hint_type] += model_info["has_answer_hint_types"][hint_type]
            
            # Calculate averages
            no_hint_avg = no_hint_success / no_hint_total if no_hint_total > 0 else 0.0
            
            # Find the best hint type for this model
            best_hint_type = None
            best_hint_avg = 0.0
            
            for hint_type in self.hint_types:
                if hint_totals[hint_type] > 0:
                    avg = hint_success[hint_type] / hint_totals[hint_type]
                    if avg > best_hint_avg:
                        best_hint_avg = avg
                        best_hint_type = hint_type
            
            # Calculate improvement
            diff = best_hint_avg - no_hint_avg
            best_hint_label = f"{best_hint_avg:.2%} ({best_hint_type})" if best_hint_type else "N/A"
            
            print(f"{tag:<20} {no_hint_avg:>15.2%} {best_hint_label:>20} {diff:>+10.2%}")

##############################################################################
# File Discovery Functions
##############################################################################

def discover_models_and_hints(results_dir: str) -> Tuple[List[str], List[str]]:
    """
    Scan the results directory to automatically discover models and hint types.
    
    Args:
        results_dir (str): Directory containing result JSON files
        
    Returns:
        Tuple[List[str], List[str]]: Lists of model tags and hint types
    """
    model_tags = set()
    hint_types = set()
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    for filename in json_files:
        # Expected format: answers_MODEL_no_hint.json or answers_MODEL_HINT_TYPE_with_hint.json
        if filename.startswith("answers_") and filename.endswith(".json"):
            parts = filename.split('_')
            
            if len(parts) >= 3:
                # Extract model name (second part)
                model_tag = parts[1]
                model_tags.add(model_tag)
                
                # Check what kind of file this is
                if "no_hint" in filename:
                    # This is a no-hint file
                    pass
                elif "with_hint" in filename and len(parts) >= 5:
                    # This is a with-hint file. The hint type might be composed of multiple parts.
                    # We need everything between the model name and "with_hint"
                    # Format: answers_MODEL_HINT_TYPE_with_hint.json
                    
                    # Find where "with_hint" begins
                    for i in range(2, len(parts)):
                        if parts[i] == "with":
                            # Extract hint type as everything between model and "with"
                            hint_type = "_".join(parts[2:i])
                            hint_types.add(hint_type)
                            break
    
    return sorted(list(model_tags)), sorted(list(hint_types))

##############################################################################
# Command-Line Interface
##############################################################################

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze inference results for multiple models.")
    parser.add_argument("--samples", "-s", type=str, required=True,
                        help="Path to original input problems (contains final_answer, difficulty_bin, llama8b_solve_rate).")
    parser.add_argument("--results_dir", "-r", type=str, default="results",
                        help="Directory where the *_no_hint.json and *_with_hint.json are located.")
    parser.add_argument("--output_dir", "-o", type=str, default="analysis_outputs",
                        help="Directory to save the plots and interesting examples.")
    parser.add_argument("--debug-extraction", "-d", action="store_true",
                        help="Enable detailed debugging for answer extraction issues.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"[Warning] Results directory '{args.results_dir}' does not exist.")
        print("Make sure you've run inference.py first to generate result files.")
        if not input("Continue anyway? (y/n): ").lower().startswith('y'):
            sys.exit(1)
    
    # Check if samples file exists
    if not os.path.exists(args.samples):
        print(f"[Error] Samples file '{args.samples}' does not exist.")
        sys.exit(1)
    
    # Discover models and hint types from result filenames
    model_tags, hint_types = discover_models_and_hints(args.results_dir)
    if not model_tags:
        print(f"[Error] No model result files found in '{args.results_dir}'.")
        sys.exit(1)
    
    print(f"[Analysis] Discovered {len(model_tags)} models: {', '.join(model_tags)}")
    print(f"[Analysis] Discovered {len(hint_types)} hint types: {', '.join(hint_types)}")

    analysis = Analysis(
        samples_file=args.samples,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        debug_extraction=args.debug_extraction
    )
    analysis.run()

if __name__ == "__main__":
    main() 