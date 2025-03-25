#!/usr/bin/env python3
"""
hint_analysis.py

Analyzes the effectiveness of individual hints across different student models,
focusing on how hints help "smart" models vs "weaker" models.

Usage:
python hint_analysis.py --samples data/joined_samples.json --results-dir results --output-dir hint_analysis --models "gemma-2-9b-it" "gemma-2-2b-it" --prompts "socratic_hint,direct_hint"
"""

import argparse
import os
import json
import sys
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from tqdm import tqdm
import csv

##############################################################################
# Helper Functions
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
    Given a result entry, return the fraction of correct answers across all samples.
    entry["correctness"] is a list of bools for each sample.
    If not found, returns 0.0.
    """
    correctness = entry.get("correctness", [])
    if not correctness:
        return 0.0
    return sum(correctness) / len(correctness)

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

def set_plot_style():
    """Set consistent plot style across all visualizations."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except ValueError:
        # Fall back to a similar style if seaborn-v0_8-whitegrid is not available
        try:
            plt.style.use('seaborn-whitegrid')
        except ValueError:
            pass  # Use default style if neither is available
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

##############################################################################
# Hint Analysis Class
##############################################################################

class HintAnalysis:
    """
    Analyzes the effectiveness of hints on student performance.
    Simplified version that:
    1. Outputs all hint data in a single file
    2. Filters hints based on configurable parameters and outputs two files
    """

    def __init__(self,
                 samples_file: str,
                 results_dir: str,
                 output_dir: str,
                 student_models: list,
                 prompt_names: list,
                 alpha: float = 0.0,  # Minimum smart improvement
                 beta: float = 1.0,   # Maximum weak improvement
                 gamma: float = 0.0): # Minimum improvement gap
        """
        Initialize the hint analysis system.
        
        Args:
            samples_file: Path to the samples file containing problems and hints
            results_dir: Directory containing result files
            output_dir: Directory to save output files
            student_models: List of student models to analyze
            prompt_names: List of prompt types to analyze
            alpha: Minimum smart improvement threshold (default: 0.0)
            beta: Maximum weak improvement threshold (default: 1.0)
            gamma: Minimum improvement gap threshold (default: 0.0)
        """
        self.samples_file = samples_file
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.student_models = student_models
        self.prompt_names = prompt_names
        
        # Filter parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Data containers
        self.problems = {}
        self.results = {}
        self.hint_data = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data
        self.load_samples()
        self.load_results()

    def run(self):
        """Run the hint analysis pipeline."""
        self.print_result_structure()
        self.evaluate_hints()
        self.save_all_hints()
        self.filter_and_save_hints()

    def load_samples(self):
        """Load problem samples from JSON file."""
        with open(self.samples_file, 'r') as f:
            samples = json.load(f)
            
        # Add problem_idx if not present in the data
        for i, problem in enumerate(samples):
            if 'problem_idx' not in problem:
                problem['problem_idx'] = i
                
            self.problems[problem['problem_idx']] = problem
            
        print(f"Loaded {len(self.problems)} problems from {self.samples_file}")

    def load_results(self):
        """Load model results from JSON files."""
        print(f"Loading results from {self.results_dir}")
        
        for model in self.student_models:
            self.results[model] = {}
            
            # Load no-hint baseline results
            baseline_path = os.path.join(self.results_dir, f"answers_{model}_no_hint.json")
            if os.path.exists(baseline_path):
                print(f"  Loading baseline results for {model}")
                try:
                    with open(baseline_path, 'r') as f:
                        self.results[model]['no_hint'] = json.load(f)
                except Exception as e:
                    print(f"  Error loading {baseline_path}: {e}")
            else:
                print(f"  Warning: No baseline results found for {model} at {baseline_path}")
            
            # Load hint results for each prompt type
            for prompt in self.prompt_names:
                hint_path = os.path.join(self.results_dir, f"answers_{model}_{prompt}_with_hint.json")
                if os.path.exists(hint_path):
                    print(f"  Loading {prompt} results for {model}")
                    try:
                        with open(hint_path, 'r') as f:
                            self.results[model][prompt] = json.load(f)
                    except Exception as e:
                        print(f"  Error loading {hint_path}: {e}")
                else:
                    print(f"  Warning: No {prompt} results found for {model} at {hint_path}")
        
        print(f"Finished loading results for {len(self.student_models)} models")

    def evaluate_hints(self):
        """
        Evaluate the effectiveness of each hint on student performance.
        Compares performance with and without hints for each model.
        
        This version extracts hints from the result files rather than the samples file.
        """
        print("\nEvaluating hint effectiveness...")
        self.hint_data = []
        
        # Debug info
        problem_count = 0
        hint_count = 0
        hint_prompt_mismatch = 0
        string_hints = 0
        dict_hints = 0
        skipped_no_text = 0
        hint_entry_count = 0
        
        # First, extract hints from the result files for each prompt
        hints_by_problem = {}  # {problem_idx: {prompt: hint_text}}
        
        print("Extracting hints from result files...")
        for model in self.student_models:
            for prompt in self.prompt_names:
                if prompt not in self.results.get(model, {}):
                    continue
                    
                prompt_results = self.results[model][prompt]
                
                if isinstance(prompt_results, list):
                    # List-based results format
                    for i, entry in enumerate(prompt_results):
                        if i >= len(self.problems):
                            break
                            
                        problem_idx = i
                        hint_text = entry.get('teacher_hint', '')
                        
                        if hint_text:
                            if problem_idx not in hints_by_problem:
                                hints_by_problem[problem_idx] = {}
                            
                            hints_by_problem[problem_idx][prompt] = hint_text
                            
                elif isinstance(prompt_results, dict):
                    # Dict-based results format
                    for problem_key, problem_data in prompt_results.items():
                        try:
                            problem_idx = int(problem_key)
                        except ValueError:
                            continue
                            
                        # Get hint text depending on the structure
                        hint_text = ''
                        if isinstance(problem_data, dict):
                            hint_text = problem_data.get('teacher_hint', '')
                        elif isinstance(problem_data, list) and problem_data:
                            hint_text = problem_data[0].get('teacher_hint', '')
                        
                        if hint_text:
                            if problem_idx not in hints_by_problem:
                                hints_by_problem[problem_idx] = {}
                            
                            hints_by_problem[problem_idx][prompt] = hint_text
        
        print(f"Found hints for {len(hints_by_problem)} problems across all prompts")
        
        # Now evaluate the extracted hints
        for problem_idx, problem in self.problems.items():
            problem_count += 1
            problem_text = problem.get('problem_text', problem.get('problem', ''))
            difficulty = problem.get('difficulty_bin', problem.get('difficulty', ''))
            
            if problem_idx not in hints_by_problem:
                continue
                
            prompt_hints = hints_by_problem[problem_idx]
            
            for prompt, hint_text in prompt_hints.items():
                hint_count += 1
                
                if not hint_text:
                    skipped_no_text += 1
                    continue
                
                for model_idx, smart_model in enumerate(self.student_models):
                    for weak_model in self.student_models:
                        if smart_model == weak_model:
                            continue
                        
                        hint_entry_count += 1
                        
                        # Get baseline performance (no hint)
                        smart_baseline = 0.0
                        weak_baseline = 0.0
                        
                        if 'no_hint' in self.results.get(smart_model, {}):
                            if isinstance(self.results[smart_model]['no_hint'], list):
                                # List format
                                if problem_idx < len(self.results[smart_model]['no_hint']):
                                    smart_baseline = get_solve_rate(self.results[smart_model]['no_hint'][problem_idx])
                            else:
                                # Dict format
                                smart_baseline = get_solve_rate(self.results[smart_model]['no_hint'].get(str(problem_idx), {}))
                        
                        if 'no_hint' in self.results.get(weak_model, {}):
                            if isinstance(self.results[weak_model]['no_hint'], list):
                                # List format
                                if problem_idx < len(self.results[weak_model]['no_hint']):
                                    weak_baseline = get_solve_rate(self.results[weak_model]['no_hint'][problem_idx])
                            else:
                                # Dict format
                                weak_baseline = get_solve_rate(self.results[weak_model]['no_hint'].get(str(problem_idx), {}))
                        
                        # Get performance with hint
                        smart_hint = 0.0
                        weak_hint = 0.0
                        
                        if prompt in self.results.get(smart_model, {}):
                            if isinstance(self.results[smart_model][prompt], list):
                                # List format
                                if problem_idx < len(self.results[smart_model][prompt]):
                                    smart_hint = get_solve_rate(self.results[smart_model][prompt][problem_idx])
                            else:
                                # Dict format
                                smart_hint = get_solve_rate(self.results[smart_model][prompt].get(str(problem_idx), {}))
                        
                        if prompt in self.results.get(weak_model, {}):
                            if isinstance(self.results[weak_model][prompt], list):
                                # List format
                                if problem_idx < len(self.results[weak_model][prompt]):
                                    weak_hint = get_solve_rate(self.results[weak_model][prompt][problem_idx])
                            else:
                                # Dict format
                                weak_hint = get_solve_rate(self.results[weak_model][prompt].get(str(problem_idx), {}))
                        
                        # Calculate improvements
                        smart_improvement = smart_hint - smart_baseline
                        weak_improvement = weak_hint - weak_baseline
                        improvement_gap = smart_improvement - weak_improvement
                        
                        # Create hint data entry
                        hint_data_entry = {
                            "problem_idx": problem_idx,
                            "problem_text": problem_text,
                            "difficulty": difficulty,
                            "hint_text": hint_text,
                            "prompt": prompt,
                            "smart_model": smart_model,
                            "weak_model": weak_model,
                            "smart_improvement": smart_improvement,
                            "weak_improvement": weak_improvement,
                            "improvement_gap": improvement_gap,
                            "avg_improvement": (smart_improvement + weak_improvement) / 2
                        }
                        
                        self.hint_data.append(hint_data_entry)
        
        # Print hint statistics
        print(f"\nHint statistics:")
        print(f"  Total problems processed: {problem_count}")
        print(f"  Total hints found: {hint_count}")
        print(f"  Total hint entries created: {len(self.hint_data)}")
        
        if not self.hint_data:
            print("\nNo hint data was created. Possible reasons:")
            print("1. The result files don't contain teacher_hint fields")
            print("2. The prompt names don't match those in your result files")
            print("3. There's a mismatch between problem indices in samples and results")
            
            print("\nTry checking:")
            print("- The structure of your result files (should have teacher_hint field)")
            print("- The prompt names used in your command match the filenames")

    def save_all_hints(self):
        """Save all hint data to a single CSV file."""
        if not self.hint_data:
            print("Warning: No hint data found to save!")
            return
            
        print(f"Saving {len(self.hint_data)} hint entries to {self.output_dir}")
        
        output_path = os.path.join(self.output_dir, "all_hints.csv")
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.hint_data[0].keys())
            writer.writeheader()
            writer.writerows(self.hint_data)
            
        # Also save as JSON for easier programmatic access
        json_path = os.path.join(self.output_dir, "all_hints.json")
        with open(json_path, 'w') as f:
            json.dump(self.hint_data, f, indent=2)
            
        print(f"Saved all hints to {output_path} and {json_path}")

    def filter_and_save_hints(self):
        """
        Filter hints based on improvement thresholds and save to separate files.
        
        Filtering criteria:
        - smart_improvement >= alpha
        - weak_improvement <= beta
        - improvement_gap >= gamma
        """
        if not self.hint_data:
            print("Warning: No hint data to filter!")
            return
            
        passing_hints = []
        failing_hints = []
        
        for hint in self.hint_data:
            if (hint["smart_improvement"] >= self.alpha and 
                hint["weak_improvement"] <= self.beta and 
                hint["improvement_gap"] >= self.gamma):
                passing_hints.append(hint)
            else:
                failing_hints.append(hint)
        
        print(f"Applied filter (alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}):")
        print(f"  Passing hints: {len(passing_hints)}/{len(self.hint_data)} ({len(passing_hints)/len(self.hint_data)*100:.1f}%)")
        print(f"  Failing hints: {len(failing_hints)}/{len(self.hint_data)} ({len(failing_hints)/len(self.hint_data)*100:.1f}%)")
        
        # Sort by improvement gap (descending)
        passing_hints.sort(key=lambda x: x["improvement_gap"], reverse=True)
        failing_hints.sort(key=lambda x: x["improvement_gap"], reverse=True)
        
        # Save passing hints
        passing_path = os.path.join(self.output_dir, "passing_hints.json")
        with open(passing_path, 'w') as f:
            json.dump(passing_hints, f, indent=2)
        
        # Save passing hints as CSV
        passing_csv = os.path.join(self.output_dir, "passing_hints.csv")
        if passing_hints:
            with open(passing_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=passing_hints[0].keys())
                writer.writeheader()
                writer.writerows(passing_hints)
        
        # Save failing hints
        failing_path = os.path.join(self.output_dir, "failing_hints.json")
        with open(failing_path, 'w') as f:
            json.dump(failing_hints, f, indent=2)
        
        # Save failing hints as CSV
        failing_csv = os.path.join(self.output_dir, "failing_hints.csv")
        if failing_hints:
            with open(failing_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=failing_hints[0].keys())
                writer.writeheader()
                writer.writerows(failing_hints)
                
        print(f"Saved filtered hints to {passing_path}, {passing_csv}, {failing_path}, and {failing_csv}")

    def print_result_structure(self):
        """Print the structure of the results files to help with debugging."""
        print("\nChecking result file structure:")
        
        # Check a few models
        for model in self.student_models[:1]:  # Just check first model
            print(f"\nModel: {model}")
            
            # Check no-hint structure
            if model in self.results and 'no_hint' in self.results[model]:
                no_hint_data = self.results[model]['no_hint']
                print(f"  No-hint data: {type(no_hint_data).__name__}")
                
                if isinstance(no_hint_data, list):
                    if no_hint_data:
                        print(f"  List with {len(no_hint_data)} entries")
                        print(f"  First entry keys: {list(no_hint_data[0].keys()) if isinstance(no_hint_data[0], dict) else 'Not a dict'}")
                elif isinstance(no_hint_data, dict):
                    print(f"  Dict with {len(no_hint_data)} keys")
                    print(f"  Keys: {list(no_hint_data.keys())}")
                    
                    # Check a sample problem
                    if no_hint_data and list(no_hint_data.keys()):
                        sample_key = list(no_hint_data.keys())[0]
                        print(f"  Sample problem '{sample_key}' entry:")
                        print(f"    Type: {type(no_hint_data[sample_key]).__name__}")
                        if isinstance(no_hint_data[sample_key], dict):
                            print(f"    Keys: {list(no_hint_data[sample_key].keys())}")
            else:
                print("  No no-hint data found")
            
            # Check hint structure for a prompt
            for prompt in self.prompt_names[:1]:  # Just check first prompt
                if model in self.results and prompt in self.results[model]:
                    hint_data = self.results[model][prompt]
                    print(f"\n  Prompt: {prompt}")
                    print(f"  Hint data: {type(hint_data).__name__}")
                    
                    if isinstance(hint_data, list):
                        if hint_data:
                            print(f"  List with {len(hint_data)} entries")
                            print(f"  First entry keys: {list(hint_data[0].keys()) if isinstance(hint_data[0], dict) else 'Not a dict'}")
                    elif isinstance(hint_data, dict):
                        print(f"  Dict with {len(hint_data)} keys")
                        print(f"  Keys: {list(hint_data.keys())}")
                        
                        # Check a sample problem and hint
                        if hint_data and list(hint_data.keys()):
                            sample_problem = list(hint_data.keys())[0]
                            print(f"  Sample problem '{sample_problem}' entry:")
                            print(f"    Type: {type(hint_data[sample_problem]).__name__}")
                            
                            if isinstance(hint_data[sample_problem], dict):
                                print(f"    Keys: {list(hint_data[sample_problem].keys())}")
                                
                                if hint_data[sample_problem] and list(hint_data[sample_problem].keys()):
                                    sample_hint = list(hint_data[sample_problem].keys())[0]
                                    print(f"    Sample hint '{sample_hint}' entry:")
                                    print(f"      Type: {type(hint_data[sample_problem][sample_hint]).__name__}")
                                    
                                    if isinstance(hint_data[sample_problem][sample_hint], dict):
                                        print(f"      Keys: {list(hint_data[sample_problem][sample_hint].keys())}")
                else:
                    print(f"\n  No data for prompt '{prompt}'")

##############################################################################
# Command-Line Interface
##############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze hint effectiveness')
    parser.add_argument('--samples', type=str, required=True, help='Path to samples file')
    parser.add_argument('--results-dir', type=str, required=True, help='Directory containing results')
    parser.add_argument('--output-dir', type=str, default='hint_analysis', help='Directory to save output')
    parser.add_argument('--models', nargs='+', required=True, help='Student models to analyze')
    parser.add_argument('--prompts', nargs='+', required=True, help='Prompt types to analyze')
    parser.add_argument('--alpha', type=float, default=0.0, help='Minimum smart improvement threshold')
    parser.add_argument('--beta', type=float, default=1.0, help='Maximum weak improvement threshold')
    parser.add_argument('--gamma', type=float, default=0.0, help='Minimum improvement gap threshold')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Handle comma-separated prompt names
    prompts = []
    for prompt_arg in args.prompts:
        # Split by comma if present
        for prompt in prompt_arg.split(','):
            if prompt.strip():
                prompts.append(prompt.strip())
    
    print(f"Running hint analysis with:")
    print(f"  Samples file: {args.samples}")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Student models: {args.models}")
    print(f"  Prompt names: {prompts}")
    print(f"  Filter parameters: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    
    analyzer = HintAnalysis(
        samples_file=args.samples,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        student_models=args.models,
        prompt_names=prompts,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    analyzer.run()

if __name__ == '__main__':
    main() 