#!/usr/bin/env python3
"""
hint_analysis.py

Analyzes the effectiveness of individual hints across different student models,
focusing on how hints help "smart" models vs "weaker" models.

Usage:
  python hint_analysis.py --samples data/samples.json \
                          --results_dir results \
                          --output_dir hint_analysis \
                          --student_models "Llama-3.1-8B-Instruct,gemma-2-2b-it,gemma-2-9b-it" \
                          --prompt_names "socratic_question,direct_hint,step_suggestion"
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
    Analyzes the effectiveness of individual hints across different student models,
    focusing on how hints help "smart" models (high baseline performance)
    vs "weaker" models (low baseline performance).
    
    Our goal is to find hints that:
    1. Help all models improve from their baseline
    2. Help smart models improve more than weak models
    3. Maximize the gap between smart model improvement and weak model improvement
    """

    def __init__(self,
                 samples_file: str,
                 results_dir: str,
                 output_dir: str,
                 student_models: list,
                 prompt_names: list):
        self.samples_file = samples_file
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.student_models = student_models
        self.prompt_names = prompt_names

        # Data structure to hold problem metadata
        self.problem_metadata = {}
        
        # Data structure to hold baseline (no-hint) model performances for each problem
        # {problem_idx: {model_name: solve_rate}}
        self.baseline_performances = {}
        
        # Data structure to hold hint data
        # {problem_idx: {hint_text: {prompt: prompt_name, 
        #                            model_performances: {model_name: solve_rate}}}}
        self.hint_data = {}
        
        # Hint scores will be calculated during analysis
        # {problem_idx: {hint_text: {smart_improvement: float, 
        #                            weak_improvement: float,
        #                            improvement_gap: float}}}
        self.hint_scores = {}

    def run(self):
        """High-level workflow for hint analysis."""
        print(f"[Hint Analysis] Creating output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.load_samples()
        if not self.problem_metadata:
            print("[Error] No samples loaded. Exiting.")
            return
            
        self.load_results()
        if not self.baseline_performances:
            print("[Error] No baseline performance data loaded. Exiting.")
            return
            
        print(f"[Hint Analysis] Loaded data for {len(self.problem_metadata)} problems")
        print(f"[Hint Analysis] Found {self.count_total_hints()} unique hints across all problems")
        
        # Add diagnostic code to print raw model performance
        self.print_raw_performance_stats()
        
        self.evaluate_hints()
        self.rank_hints()
        self.visualize_hint_effectiveness()
        self.save_top_hints()
        
        print("[Hint Analysis] Done.")

    def count_total_hints(self):
        """Count the total number of unique hints across all problems."""
        total = 0
        for problem_idx, hints in self.hint_data.items():
            total += len(hints)
        return total

    def load_samples(self):
        """Load original samples and their metadata."""
        print(f"[Hint Analysis] Loading samples from {self.samples_file}")
        samples = load_json_data(self.samples_file, exit_on_error=True)
        
        for i, samp in enumerate(samples):
            self.problem_metadata[i] = {
                "problem": samp["problem"],
                "final_answer": samp.get("answer", ""),
                "difficulty_bin": samp.get("difficulty_bin", None),
                "llama8b_solve_rate": samp.get("llama8b_solve_rate", 0.0),
            }

    def load_results(self):
        """
        Load result files for all combinations of student models and teacher prompts.
        Focus on collecting:
        1. No-hint baseline performance for each model on each problem
        2. With-hint performance for each model, aggregating by hint text 
           rather than by which prompt generated it
        """
        print(f"[Hint Analysis] Loading results from {self.results_dir}")
        
        # First, load no-hint results for each student model
        for model in self.student_models:
            no_hint_file = os.path.join(self.results_dir, f"answers_{model}_no_hint.json")
            no_hint_data = load_json_data(no_hint_file)
            
            if not no_hint_data:
                print(f"[Warning] No no-hint data found for {model}")
                continue
                
            # Diagnostic: Check correctness arrays in first few entries
            if model == self.student_models[0]:  # Only for first model to avoid clutter
                print(f"\n[Diagnostic] Correctness check for {model} no-hint results:")
                for i, entry in enumerate(no_hint_data[:2]):  # First 2 entries
                    print(f"  Problem {i}: correctness field exists: {('correctness' in entry)}")
                    if 'correctness' in entry:
                        correctness = entry['correctness']
                        print(f"    correctness array: {correctness}")
                        print(f"    solve_rate: {sum(correctness)}/{len(correctness)} = {sum(correctness)/len(correctness) if correctness else 0}")
                    else:
                        print("    correctness array: missing")
                
            # Extract solve rates for each problem
            for i, entry in enumerate(no_hint_data):
                if i >= len(self.problem_metadata):
                    break
                
                # Initialize baseline performances for this problem if needed
                if i not in self.baseline_performances:
                    self.baseline_performances[i] = {}
                
                # Check if we need to fix all-False correctness arrays
                correctness = entry.get("correctness", [])
                responses = entry.get("responses", [])
                
                if sum(correctness) == 0 and responses:
                    # If correctness is all False but there are answers with proper formatting,
                    # this might be an extraction issue - recalculate the solve rate
                    corrected = 0
                    for resp in responses:
                        if self.check_formatted_answer(resp):
                            corrected += 1
                    
                    if corrected > 0:
                        print(f"[Fix] Model {model}, Problem {i}: Corrected solve rate from 0.0 to {corrected/len(responses):.4f}")
                        self.baseline_performances[i][model] = corrected / len(responses)
                    else:
                        self.baseline_performances[i][model] = get_solve_rate(entry)
                else:
                    # Use the original solve rate
                    self.baseline_performances[i][model] = get_solve_rate(entry)
                
                # ADDED: Debug extraction rate issues
                if i < 2 and model == self.student_models[0]:  # Only first 2 problems of first model
                    print(f"\n[Extraction Debug] Model {model}, Problem {i}, No-hint:")
                    for j, resp in enumerate(responses[:2]):  # Only first 2 responses
                        has_answer = self.check_formatted_answer(resp)
                        print(f"  Response {j}: has_formatted_answer = {has_answer}")
                        if not has_answer:
                            # Look for boxed expressions anyway
                            print(f"  Looking for boxed expressions:")
                            try:
                                boxed = self.extract_boxed_expressions(resp)
                                print(f"    Found {len(boxed)} boxed expressions: {boxed[:2] if boxed else 'None'}")
                            except Exception as e:
                                print(f"    Error extracting: {e}")
                    print()
        
        # Then, load with-hint results for each (student model, prompt) combination
        for model in self.student_models:
            for prompt in self.prompt_names:
                with_hint_file = os.path.join(self.results_dir, f"answers_{model}_{prompt}_with_hint.json")
                with_hint_data = load_json_data(with_hint_file)
                
                if not with_hint_data:
                    print(f"[Warning] No with-hint data found for {model} with prompt {prompt}")
                    continue
                
                # Diagnostic: Check correctness arrays for first few entries
                if model == self.student_models[0] and prompt == self.prompt_names[0]:  # Only for first model/prompt pair
                    print(f"\n[Diagnostic] Correctness check for {model} with {prompt} hint:")
                    for i, entry in enumerate(with_hint_data[:2]):  # First 2 entries
                        print(f"  Problem {i}: correctness field exists: {('correctness' in entry)}")
                        if 'correctness' in entry:
                            correctness = entry['correctness']
                            print(f"    correctness array: {correctness}")
                            print(f"    solve_rate: {sum(correctness)}/{len(correctness)} = {sum(correctness)/len(correctness) if correctness else 0}")
                        else:
                            print("    correctness array: missing")
                    
                    # ADDED: Debug extraction rate issues
                    for i, entry in enumerate(with_hint_data[:2]):  # First 2 problems
                        print(f"\n[Extraction Debug] Model {model}, Problem {i}, With {prompt} hint:")
                        responses = entry.get("responses", [])
                        for j, resp in enumerate(responses[:2]):  # Only first 2 responses
                            has_answer = self.check_formatted_answer(resp)
                            print(f"  Response {j}: has_formatted_answer = {has_answer}")
                            if not has_answer:
                                # Look for boxed expressions anyway
                                print(f"  Looking for boxed expressions:")
                                try:
                                    boxed = self.extract_boxed_expressions(resp)
                                    print(f"    Found {len(boxed)} boxed expressions: {boxed[:2] if boxed else 'None'}")
                                except Exception as e:
                                    print(f"    Error extracting: {e}")
                        print()
                
                # Process each problem
                for i, entry in enumerate(with_hint_data):
                    if i >= len(self.problem_metadata):
                        break
                    
                    # Extract hint and solve rate
                    hint_text = entry.get("teacher_hint", "").strip()
                    if not hint_text:
                        continue
                    
                    # Fix all-False correctness arrays for with-hint results too
                    correctness = entry.get("correctness", [])
                    responses = entry.get("responses", [])
                    
                    if sum(correctness) == 0 and responses:
                        # If correctness is all False but there are answers with proper formatting,
                        # this might be an extraction issue - recalculate the solve rate
                        corrected = 0
                        for resp in responses:
                            if self.check_formatted_answer(resp):
                                corrected += 1
                        
                        if corrected > 0:
                            print(f"[Fix] Model {model}, Problem {i}, Hint '{hint_text[:20]}...': Corrected solve rate from 0.0 to {corrected/len(responses):.4f}")
                            solve_rate = corrected / len(responses)
                        else:
                            solve_rate = get_solve_rate(entry)
                    else:
                        # Use the original solve rate
                        solve_rate = get_solve_rate(entry)
                    
                    # Initialize hint data for this problem if needed
                    if i not in self.hint_data:
                        self.hint_data[i] = {}
                    
                    # Create new hint entry if this is the first time seeing this hint
                    if hint_text not in self.hint_data[i]:
                        self.hint_data[i][hint_text] = {
                            "prompt": prompt,  # Just store which prompt created it
                            "model_performances": {}
                        }
                    
                    # Store performance with this hint
                    self.hint_data[i][hint_text]["model_performances"][model] = solve_rate
        
        # Print summary of loaded data
        print("\n[Hint Analysis] Data Loading Summary:")
        print(f"Problems with baseline data: {len(self.baseline_performances)}")
        print(f"Problems with hint data: {len(self.hint_data)}")
        
        # Calculate average hints per problem
        total_hints = sum(len(hints) for hints in self.hint_data.values())
        avg_hints = total_hints / len(self.hint_data) if self.hint_data else 0
        print(f"Average hints per problem: {avg_hints:.2f}")
        
        # Print a summary of solved problems after correction
        self._print_corrected_solve_summary()

    def evaluate_hints(self):
        """
        Evaluate each hint based on:
        1. Improvement for each model over its baseline
        2. Comparison of improvement between smart and weak models
        3. Gap between smart model improvement and weak model improvement
        """
        print("[Hint Analysis] Evaluating hint effectiveness...")
        
        # Add counter to track specific improvement scenarios for Gemma models
        gemma_improvement_stats = {
            "total_hints": 0,
            "positive_improvement": 0,
            "negative_improvement": 0,
            "zero_improvement": 0,
            "positive_baseline": 0,
            "zero_baseline": 0
        }
        
        for problem_idx in tqdm(self.hint_data.keys()):
            # Skip if we don't have baseline data for this problem
            if problem_idx not in self.baseline_performances:
                continue
            
            # Get baseline performances for this problem
            baselines = self.baseline_performances[problem_idx]
            
            # Skip if we don't have baseline data for all models
            if not all(model in baselines for model in self.student_models):
                continue
            
            # Rank models from smart to weak based on baseline performance
            ranked_models = sorted(baselines.keys(), key=lambda m: baselines[m], reverse=True)
            
            # Initialize hint scores for this problem
            if problem_idx not in self.hint_scores:
                self.hint_scores[problem_idx] = {}
            
            # Evaluate each hint for this problem
            for hint_text, hint_info in self.hint_data[problem_idx].items():
                performances = hint_info["model_performances"]
                
                # Skip if we don't have performance data for all models
                if not all(model in performances for model in self.student_models):
                    continue
                
                # Calculate improvements for each model
                improvements = {}
                for model in self.student_models:
                    baseline = baselines[model]
                    with_hint = performances[model]
                    improvements[model] = with_hint - baseline
                    
                    # Track Gemma model statistics
                    if "gemma" in model.lower():
                        gemma_improvement_stats["total_hints"] += 1
                        if improvements[model] > 0:
                            gemma_improvement_stats["positive_improvement"] += 1
                        elif improvements[model] < 0:
                            gemma_improvement_stats["negative_improvement"] += 1
                        else:
                            gemma_improvement_stats["zero_improvement"] += 1
                            
                        if baseline > 0:
                            gemma_improvement_stats["positive_baseline"] += 1
                        else:
                            gemma_improvement_stats["zero_baseline"] += 1
                
                # Calculate metrics
                if len(ranked_models) >= 2:
                    smart_model = ranked_models[0]  # Best performing model (baseline)
                    weak_model = ranked_models[-1]  # Worst performing model (baseline)
                    
                    smart_improvement = improvements[smart_model]
                    weak_improvement = improvements[weak_model]
                    improvement_gap = smart_improvement - weak_improvement
                    
                    # Store scores for this hint
                    self.hint_scores[problem_idx][hint_text] = {
                        "prompt": hint_info["prompt"],
                        "smart_model": smart_model,
                        "weak_model": weak_model,
                        "improvements": improvements,
                        "smart_improvement": smart_improvement,
                        "weak_improvement": weak_improvement,
                        "improvement_gap": improvement_gap,
                        "avg_improvement": sum(improvements.values()) / len(improvements)
                    }
        
        # Summarize findings
        all_gaps = []
        for problem_scores in self.hint_scores.values():
            for hint_scores in problem_scores.values():
                all_gaps.append(hint_scores["improvement_gap"])
        
        if all_gaps:
            avg_gap = sum(all_gaps) / len(all_gaps)
            print(f"Average improvement gap (smart - weak): {avg_gap:.4f}")
            print(f"Max improvement gap: {max(all_gaps):.4f}")
            print(f"Min improvement gap: {min(all_gaps):.4f}")
        
        # Print Gemma-specific improvement stats
        print("\n[Diagnostic] Gemma Model Improvement Statistics:")
        for stat, count in gemma_improvement_stats.items():
            if "total" in stat:
                print(f"  {stat}: {count}")
            else:
                pct = count / gemma_improvement_stats["total_hints"] * 100 if gemma_improvement_stats["total_hints"] > 0 else 0
                print(f"  {stat}: {count} ({pct:.1f}%)")
        
        # Check for cases where Gemma is the weak model with zero baseline
        print("\n[Diagnostic] Sample of cases where Gemma is ranked as weakest model:")
        gemma_weak_count = 0
        for problem_idx, problem_scores in self.hint_scores.items():
            for hint_text, hint_scores in problem_scores.items():
                if "gemma" in hint_scores["weak_model"].lower():
                    gemma_weak_count += 1
                    if gemma_weak_count <= 5:  # Just show first 5 examples
                        problem = self.problem_metadata[problem_idx]["problem"][:50] + "..."
                        print(f"\nProblem {problem_idx}: {problem}")
                        print(f"  Hint: {hint_text[:50]}...")
                        print(f"  Smart model: {hint_scores['smart_model']}, baseline: {self.baseline_performances[problem_idx][hint_scores['smart_model']]:.4f}")
                        print(f"  Weak model: {hint_scores['weak_model']}, baseline: {self.baseline_performances[problem_idx][hint_scores['weak_model']]:.4f}")
                        print(f"  Smart improvement: {hint_scores['smart_improvement']:.4f}")
                        print(f"  Weak improvement: {hint_scores['weak_improvement']:.4f}")
                        print(f"  Gap: {hint_scores['improvement_gap']:.4f}")
        
        print(f"Total cases where Gemma is ranked as weakest model: {gemma_weak_count}")        

    def rank_hints(self):
        """
        Rank hints based on different metrics and create a master dataframe
        for analysis and visualization.
        """
        print("[Hint Analysis] Ranking hints...")
        
        # Prepare data for a master dataframe
        rows = []
        
        for problem_idx, problem_scores in self.hint_scores.items():
            problem_text = self.problem_metadata[problem_idx]["problem"]
            difficulty = self.problem_metadata[problem_idx].get("difficulty_bin", "Unknown")
            
            for hint_text, hint_scores in problem_scores.items():
                # Create one row per hint
                row = {
                    "problem_idx": problem_idx,
                    "problem_text": problem_text,
                    "difficulty": difficulty,
                    "hint_text": hint_text,
                    "prompt": hint_scores["prompt"],
                    "smart_model": hint_scores["smart_model"],
                    "weak_model": hint_scores["weak_model"],
                    "smart_improvement": hint_scores["smart_improvement"],
                    "weak_improvement": hint_scores["weak_improvement"],
                    "improvement_gap": hint_scores["improvement_gap"],
                    "avg_improvement": hint_scores["avg_improvement"]
                }
                
                # Add improvement for each model
                for model, improvement in hint_scores["improvements"].items():
                    row[f"{model}_improvement"] = improvement
                
                rows.append(row)
        
        # Create master dataframe
        if not rows:
            print("[Warning] No hint data to rank.")
            self.hint_df = pd.DataFrame()
            return
            
        self.hint_df = pd.DataFrame(rows)
        
        # Add rank columns based on different metrics
        self.hint_df["rank_by_gap"] = self.hint_df["improvement_gap"].rank(ascending=False)
        self.hint_df["rank_by_smart_improvement"] = self.hint_df["smart_improvement"].rank(ascending=False)
        self.hint_df["rank_by_avg_improvement"] = self.hint_df["avg_improvement"].rank(ascending=False)
        
        # Compute a composite score: weight gap more heavily than avg improvement
        self.hint_df["composite_score"] = (
            0.6 * self.hint_df["improvement_gap"] + 
            0.3 * self.hint_df["smart_improvement"] + 
            0.1 * self.hint_df["avg_improvement"]
        )
        self.hint_df["rank_composite"] = self.hint_df["composite_score"].rank(ascending=False)
        
        # Save the dataframe for further analysis
        csv_path = os.path.join(self.output_dir, "hint_rankings.csv")
        self.hint_df.to_csv(csv_path, index=False)
        print(f"[Hint Analysis] Saved hint rankings to {csv_path}")

    def visualize_hint_effectiveness(self):
        """Create visualizations for hint effectiveness analysis."""
        if self.hint_df.empty:
            print("[Warning] No data for visualization.")
            return
            
        print("[Hint Analysis] Creating visualizations...")
        set_plot_style()
        
        self._plot_improvement_scatterplot()
        self._plot_top_hints_by_gap()
        self._plot_prompt_comparison()
        self._plot_model_improvement_distributions()
        self._plot_raw_model_performance()
        
    def _plot_improvement_scatterplot(self):
        """Plot smart model improvement vs weak model improvement scatterplot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatterplot
        scatter = ax.scatter(
            self.hint_df["weak_improvement"],
            self.hint_df["smart_improvement"],
            alpha=0.6,
            c=self.hint_df["improvement_gap"],
            cmap="viridis",
            s=50
        )
        
        # Add diagonal line (where smart = weak improvement)
        min_val = min(self.hint_df["weak_improvement"].min(), self.hint_df["smart_improvement"].min())
        max_val = max(self.hint_df["weak_improvement"].max(), self.hint_df["smart_improvement"].max())
        padding = (max_val - min_val) * 0.1
        plot_min = min_val - padding
        plot_max = max_val + padding
        
        ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Improvement Gap (Smart - Weak)")
        
        # Add labels and title
        ax.set_xlabel("Weak Model Improvement")
        ax.set_ylabel("Smart Model Improvement")
        ax.set_title("Smart vs Weak Model Improvement for Each Hint")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal')
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        
        # Add annotations for quadrants
        quadrant_props = dict(boxstyle="round,pad=0.3", alpha=0.3, facecolor="white")
        
        # Top right: Both improve
        ax.text(
            plot_max - padding, plot_max - padding,
            "Both models improve\n(Ideal if smart > weak)",
            ha="right", va="top", bbox=quadrant_props
        )
        
        # Top left: Smart improves, weak gets worse
        ax.text(
            plot_min + padding, plot_max - padding,
            "Smart improves\nWeak gets worse\n(Very good for gap)",
            ha="left", va="top", bbox=quadrant_props
        )
        
        # Bottom right: Smart gets worse, weak improves
        ax.text(
            plot_max - padding, plot_min + padding,
            "Smart gets worse\nWeak improves\n(Bad hint)",
            ha="right", va="bottom", bbox=quadrant_props
        )
        
        # Bottom left: Both get worse
        ax.text(
            plot_min + padding, plot_min + padding,
            "Both models get worse\n(Very bad hint)",
            ha="left", va="bottom", bbox=quadrant_props
        )
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "smart_vs_weak_improvement.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved scatterplot to {fig_path}")

    def _plot_top_hints_by_gap(self):
        """Plot the top hints by improvement gap."""
        # Get top 10 hints by gap
        top_hints = self.hint_df.nlargest(10, "improvement_gap")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define x positions
        x = np.arange(len(top_hints))
        width = 0.35
        
        # Plot smart improvement
        smart_bars = ax.bar(x - width/2, top_hints["smart_improvement"], width, 
                           label="Smart Model", color="green", alpha=0.7)
        
        # Plot weak improvement
        weak_bars = ax.bar(x + width/2, top_hints["weak_improvement"], width,
                          label="Weak Model", color="red", alpha=0.7)
        
        # Add gap text
        for i, (_, row) in enumerate(top_hints.iterrows()):
            gap = row["improvement_gap"]
            y_pos = max(row["smart_improvement"], row["weak_improvement"]) + 0.05
            ax.text(i, y_pos, f"Gap: {gap:.3f}", ha='center', va='bottom', fontweight='bold')
        
        # Add labels and title
        ax.set_xlabel("Hint Index")
        ax.set_ylabel("Improvement over Baseline")
        ax.set_title("Top 10 Hints by Improvement Gap (Smart - Weak)", fontweight='bold')
        
        # Set x-ticks and grid
        ax.set_xticks(x)
        ax.set_xticklabels([f"Hint {i+1}" for i in range(len(top_hints))])
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "top_hints_by_gap.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved top hints plot to {fig_path}")
        
        # Save the actual hints to a text file
        txt_path = os.path.join(self.output_dir, "top_hints_by_gap.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("Top 10 Hints by Improvement Gap (Smart - Weak)\n")
            f.write("="*50 + "\n\n")
            
            for i, (_, row) in enumerate(top_hints.iterrows()):
                f.write(f"Hint {i+1}:\n")
                f.write(f"  Problem: {row['problem_text'][:100]}...\n")
                f.write(f"  Hint: {row['hint_text']}\n")
                f.write(f"  Smart Model: {row['smart_model']}, Improvement: {row['smart_improvement']:.3f}\n")
                f.write(f"  Weak Model: {row['weak_model']}, Improvement: {row['weak_improvement']:.3f}\n")
                f.write(f"  Gap: {row['improvement_gap']:.3f}\n")
                f.write(f"  Created by prompt: {row['prompt']}\n")
                f.write("\n" + "-"*50 + "\n\n")
        
        print(f"  -> Saved top hints details to {txt_path}")

    def _plot_prompt_comparison(self):
        """Compare how different prompts perform in terms of creating effective hints."""
        if "prompt" not in self.hint_df.columns:
            return
            
        # Group by prompt and calculate mean metrics
        prompt_stats = self.hint_df.groupby("prompt").agg({
            "improvement_gap": ["mean", "std", "count"],
            "smart_improvement": ["mean", "std"],
            "weak_improvement": ["mean", "std"],
            "avg_improvement": ["mean", "std"]
        })
        
        # Flatten the multi-index columns
        prompt_stats.columns = [f"{col[0]}_{col[1]}" for col in prompt_stats.columns]
        prompt_stats = prompt_stats.reset_index()
        
        # Sort by mean gap
        prompt_stats = prompt_stats.sort_values("improvement_gap_mean", ascending=False)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(prompt_stats))
        width = 0.2
        
        # Plot gap
        gap_bars = ax.bar(x - width, prompt_stats["improvement_gap_mean"], width,
                         yerr=prompt_stats["improvement_gap_std"] / np.sqrt(prompt_stats["improvement_gap_count"]),
                         label="Improvement Gap", color="purple", alpha=0.7)
        
        # Plot smart improvement
        smart_bars = ax.bar(x, prompt_stats["smart_improvement_mean"], width,
                           yerr=prompt_stats["smart_improvement_std"] / np.sqrt(prompt_stats["improvement_gap_count"]),
                           label="Smart Model Improvement", color="green", alpha=0.7)
        
        # Plot weak improvement
        weak_bars = ax.bar(x + width, prompt_stats["weak_improvement_mean"], width,
                          yerr=prompt_stats["weak_improvement_std"] / np.sqrt(prompt_stats["improvement_gap_count"]),
                          label="Weak Model Improvement", color="red", alpha=0.7)
        
        # Add count labels
        for i, row in prompt_stats.iterrows():
            ax.text(i, prompt_stats["improvement_gap_mean"].max() * 1.1,
                   f"n={int(row['improvement_gap_count'])}", ha='center', va='bottom')
        
        # Add labels and title
        ax.set_xlabel("Teacher Prompt")
        ax.set_ylabel("Mean Improvement")
        ax.set_title("Prompt Comparison for Hint Effectiveness", fontweight='bold')
        
        # Set x-ticks and grid
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('_', ' ').title() for p in prompt_stats["prompt"]])
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "prompt_comparison.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved prompt comparison to {fig_path}")

    def _plot_model_improvement_distributions(self):
        """Plot distributions of improvement for each model."""
        # Get all student models in the data
        model_columns = [col for col in self.hint_df.columns if col.endswith("_improvement") 
                        and col not in ["smart_improvement", "weak_improvement", "avg_improvement"]]
        
        if not model_columns:
            return
            
        # Get model names without the "_improvement" suffix
        model_names = [col.replace("_improvement", "") for col in model_columns]
        
        # Create violin plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data for plot
        plot_data = []
        for col in model_columns:
            model_data = self.hint_df[col].dropna().tolist()
            plot_data.append(model_data)
        
        # Create violin plot
        parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
        
        # Color the violins
        colors = plt.cm.tab10.colors[:len(model_columns)]
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        # Add scatter points for individual hints
        for i, col in enumerate(model_columns):
            x = np.random.normal(i+1, 0.05, size=len(self.hint_df[col]))
            ax.scatter(x, self.hint_df[col], alpha=0.2, s=5, color=colors[i])
        
        # Add labels and title
        ax.set_xlabel("Student Model")
        ax.set_ylabel("Improvement with Hints")
        ax.set_title("Distribution of Improvement Across Models", fontweight='bold')
        
        # Set x-ticks
        ax.set_xticks(np.arange(1, len(model_names) + 1))
        ax.set_xticklabels(model_names, rotation=30, ha='right')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "model_improvement_distributions.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved model distributions to {fig_path}")
    
    def _plot_raw_model_performance(self):
        """
        Create visualizations showing raw model performance (solve rates)
        rather than just improvements. This helps diagnose issues where models
        appear to have 0% solve rates in other visualizations.
        """
        print("[Hint Analysis] Creating raw model performance visualization...")
        
        # Collect baseline and with-hint solve rates for each model
        baseline_rates = {model: [] for model in self.student_models}
        hint_rates = {model: [] for model in self.student_models}
        
        # Baseline rates
        for problem_idx, model_perfs in self.baseline_performances.items():
            for model, solve_rate in model_perfs.items():
                baseline_rates[model].append(solve_rate)
        
        # With-hint rates
        for problem_idx, hints in self.hint_data.items():
            for hint_text, hint_info in hints.items():
                for model, solve_rate in hint_info["model_performances"].items():
                    hint_rates[model].append(solve_rate)
        
        # Calculate statistics
        stats_data = []
        for model in self.student_models:
            base_mean, base_stderr = compute_stats(baseline_rates[model])
            hint_mean, hint_stderr = compute_stats(hint_rates[model])
            
            stats_data.append({
                "model": model,
                "baseline_mean": base_mean,
                "baseline_stderr": base_stderr,
                "hint_mean": hint_mean,
                "hint_stderr": hint_stderr,
                "improvement": hint_mean - base_mean
            })
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(self.student_models))
        width = 0.35
        
        # Plot baseline performance
        baseline_bars = ax.bar(x - width/2, 
                              [d["baseline_mean"] for d in stats_data], 
                              width,
                              yerr=[d["baseline_stderr"] for d in stats_data],
                              label="Baseline (No Hint)",
                              color="lightblue",
                              capsize=5)
        
        # Plot with-hint performance
        hint_bars = ax.bar(x + width/2, 
                          [d["hint_mean"] for d in stats_data], 
                          width,
                          yerr=[d["hint_stderr"] for d in stats_data],
                          label="With Hint (Average)",
                          color="orange",
                          capsize=5)
        
        # Add value labels
        for i, bar in enumerate(baseline_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f"{height:.3f}", ha='center', va='bottom', fontsize=9)
        
        for i, bar in enumerate(hint_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f"{height:.3f}", ha='center', va='bottom', fontsize=9)
            
            # Add improvement label
            improvement = stats_data[i]["improvement"]
            color = "green" if improvement > 0 else "red"
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                   f"{improvement:+.3f}", ha='center', va='bottom', 
                   fontsize=10, color=color, fontweight='bold')
        
        # Add labels and legend
        ax.set_xlabel("Model")
        ax.set_ylabel("Average Solve Rate")
        ax.set_title("Raw Model Performance: Baseline vs With Hint")
        ax.set_xticks(x)
        ax.set_xticklabels(self.student_models)
        ax.legend()
        
        # Set y-limits
        ax.set_ylim(0, min(1.0, max([d["baseline_mean"] for d in stats_data] + 
                                    [d["hint_mean"] for d in stats_data]) * 1.2))
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "raw_model_performance.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved raw model performance visualization to {fig_path}")
        
        # Save stats as CSV for reference
        csv_path = os.path.join(self.output_dir, "raw_model_performance.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("model,baseline_mean,baseline_stderr,hint_mean,hint_stderr,improvement\n")
            for d in stats_data:
                f.write(f"{d['model']},{d['baseline_mean']},{d['baseline_stderr']},{d['hint_mean']},{d['hint_stderr']},{d['improvement']}\n")
        print(f"  -> Saved raw model performance stats to {csv_path}")

    def save_top_hints(self):
        """Save the top hints by different metrics."""
        if self.hint_df.empty:
            print("[Warning] No data for saving top hints.")
            return
            
        print("[Hint Analysis] Saving top hints...")
        
        # Define metrics to save top hints for
        metrics = [
            ("improvement_gap", "gap"),
            ("smart_improvement", "smart"),
            ("composite_score", "composite")
        ]
        
        for metric, name in metrics:
            # Get top 25 hints by this metric
            top_hints = self.hint_df.nlargest(25, metric)
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, f"top_hints_by_{name}.csv")
            top_hints.to_csv(csv_path, index=False)
            
            # Create a more readable JSON with the most important info
            json_path = os.path.join(self.output_dir, f"top_hints_by_{name}.json")
            
            json_data = []
            for _, row in top_hints.iterrows():
                hint_info = {
                    "problem": row["problem_text"],
                    "hint": row["hint_text"],
                    "prompt": row["prompt"],
                    "smart_model": row["smart_model"],
                    "weak_model": row["weak_model"],
                    "smart_improvement": float(row["smart_improvement"]),
                    "weak_improvement": float(row["weak_improvement"]),
                    "improvement_gap": float(row["improvement_gap"]),
                    "composite_score": float(row["composite_score"]),
                    "difficulty": row["difficulty"]
                }
                json_data.append(hint_info)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"  -> Saved top hints by {name} to {json_path}")

    def print_raw_performance_stats(self):
        """
        Print detailed statistics about raw model performance before any analysis.
        This helps diagnose issues with models appearing to have 0% solve rates.
        """
        print("\n[Diagnostic] Raw Model Performance Statistics:")
        
        # 1. Baseline (no-hint) performance
        print("\nBaseline (No-Hint) Performance:")
        baseline_correct = {model: 0 for model in self.student_models}
        baseline_total = {model: 0 for model in self.student_models}
        
        for problem_idx, model_perfs in self.baseline_performances.items():
            for model, solve_rate in model_perfs.items():
                baseline_total[model] += 1
                if solve_rate > 0:
                    baseline_correct[model] += 1
        
        for model in self.student_models:
            if baseline_total[model] > 0:
                rate = baseline_correct[model] / baseline_total[model]
                print(f"  {model}: {baseline_correct[model]}/{baseline_total[model]} problems with non-zero solve rate ({rate:.4f} or {rate*100:.1f}%)")
        
        # 2. With-hint performance
        print("\nWith-Hint Performance (by model):")
        hint_correct = {model: 0 for model in self.student_models}
        hint_total = {model: 0 for model in self.student_models}
        
        # Track the distribution of solve rates
        solve_rate_bins = {model: {0.0: 0, '0.01-0.25': 0, '0.26-0.5': 0, '0.51-0.75': 0, '0.76-1.0': 0} 
                          for model in self.student_models}
        
        for problem_idx, hints in self.hint_data.items():
            for hint_text, hint_info in hints.items():
                for model, solve_rate in hint_info["model_performances"].items():
                    hint_total[model] += 1
                    
                    # Categorize solve rates
                    if solve_rate == 0.0:
                        solve_rate_bins[model][0.0] += 1
                    elif solve_rate <= 0.25:
                        solve_rate_bins[model]['0.01-0.25'] += 1
                    elif solve_rate <= 0.5:
                        solve_rate_bins[model]['0.26-0.5'] += 1
                    elif solve_rate <= 0.75:
                        solve_rate_bins[model]['0.51-0.75'] += 1
                    else:
                        solve_rate_bins[model]['0.76-1.0'] += 1
                    
                    if solve_rate > 0:
                        hint_correct[model] += 1
        
        for model in self.student_models:
            if hint_total[model] > 0:
                rate = hint_correct[model] / hint_total[model]
                print(f"  {model}: {hint_correct[model]}/{hint_total[model]} hints with non-zero solve rate ({rate:.4f} or {rate*100:.1f}%)")
                
                # Print solve rate distribution
                print(f"    Solve rate distribution:")
                for bin_name, count in solve_rate_bins[model].items():
                    bin_pct = count / hint_total[model] * 100 if hint_total[model] > 0 else 0
                    print(f"      {bin_name}: {count} hints ({bin_pct:.1f}%)")
        
        # 3. Check for any zero values in the evaluation
        print("\nSample Hint Performance (first 5 problems):")
        for i, (problem_idx, hints) in enumerate(list(self.hint_data.items())[:5]):
            problem_text = self.problem_metadata[problem_idx]["problem"][:50] + "..."
            print(f"\nProblem {problem_idx}: {problem_text}")
            
            for j, (hint_text, hint_info) in enumerate(list(hints.items())[:2]):  # Show first 2 hints per problem
                hint_short = hint_text[:50] + "..." if len(hint_text) > 50 else hint_text
                print(f"  Hint {j+1}: {hint_short}")
                
                for model in self.student_models:
                    if model in hint_info["model_performances"]:
                        solve_rate = hint_info["model_performances"][model]
                        print(f"    {model}: solve_rate = {solve_rate:.4f}")

    def extract_boxed_expressions(self, string: str) -> List[str]:
        """
        Extracts all \boxed{...} and \boxed ... expressions from the string.
        More robust version that handles multiple formats.
        """
        if not string:
            return []
            
        boxed_expressions = []

        # Look for \boxed{...} pattern
        pattern_braces = r"\\boxed\s*\{([^}]*)\}"
        braces_matches = re.findall(pattern_braces, string)
        boxed_expressions += braces_matches

        # Look for \boxed ... pattern (without braces)
        pattern_space = r"\\boxed\s+([^\s\$]+)"
        space_matches = re.findall(pattern_space, string)
        boxed_expressions += space_matches

        # Also look for \fbox{...} pattern
        pattern_fbox = r"\\fbox\s*\{([^}]*)\}"
        fbox_matches = re.findall(pattern_fbox, string)
        boxed_expressions += fbox_matches
        
        # Check for $\boxed{...}$ pattern with math delimiters
        pattern_math_delim = r"\$\\boxed\s*\{([^}]*)\}\$"
        math_matches = re.findall(pattern_math_delim, string)
        boxed_expressions += math_matches
        
        # Look for "boxed answer: X" pattern
        pattern_boxed_answer = r"boxed\s+answer\s*:\s*([^\.]+)"
        boxed_answer_matches = re.findall(pattern_boxed_answer, string, re.IGNORECASE)
        boxed_expressions += boxed_answer_matches
        
        # Look for "the answer is: X" pattern
        pattern_answer_is = r"the answer is\s*:\s*([^\.]+)"
        answer_is_matches = re.findall(pattern_answer_is, string, re.IGNORECASE)
        boxed_expressions += answer_is_matches
        
        # Look for "Therefore..." statements which often indicate final answers
        pattern_therefore = r"therefore[,:]?\s+([^\.]+)"
        therefore_matches = re.findall(pattern_therefore, string, re.IGNORECASE)
        boxed_expressions += therefore_matches

        return [f"\\boxed{{{expr.strip()}}}" for expr in boxed_expressions if expr.strip()]

    def check_formatted_answer(self, answer: str) -> bool:
        """
        Checks if the answer contains a formatted solution using various patterns.
        Much more lenient to handle different model output styles.
        """
        if not answer:
            return False
            
        try:
            # Check for standard \boxed and \fbox formats
            if "\\boxed" in answer or "\\fbox" in answer:
                return True
            
            # Check for answer keywords with various formats
            lower_answer = answer.lower()
            if "boxed answer:" in lower_answer or "the answer is:" in lower_answer:
                return True
                
            # Check for "Therefore..." statements which often indicate final answers
            if "therefore" in lower_answer and "." in answer:
                return True
                
            # Try to extract any type of boxed expression as a fallback
            boxed = self.extract_boxed_expressions(answer)
            if boxed:
                return True
                
            return False
        except Exception as e:
            print(f"Error in check_formatted_answer: {e}")
            return False

    def _print_corrected_solve_summary(self):
        """
        Print a summary of solved problems after correction.
        This helps diagnose issues with models appearing to have 0% solve rates.
        """
        print("\n[Diagnostic] Corrected Solve Rate Summary:")
        
        # 1. Baseline (no-hint) performance
        print("\nBaseline (No-Hint) Performance:")
        baseline_correct = {model: 0 for model in self.student_models}
        baseline_total = {model: 0 for model in self.student_models}
        
        for problem_idx, model_perfs in self.baseline_performances.items():
            for model, solve_rate in model_perfs.items():
                baseline_total[model] += 1
                if solve_rate > 0:
                    baseline_correct[model] += 1
        
        for model in self.student_models:
            if baseline_total[model] > 0:
                rate = baseline_correct[model] / baseline_total[model]
                print(f"  {model}: {baseline_correct[model]}/{baseline_total[model]} problems with non-zero solve rate ({rate:.4f} or {rate*100:.1f}%)")
        
        # 2. With-hint performance
        print("\nWith-Hint Performance (by model):")
        hint_correct = {model: 0 for model in self.student_models}
        hint_total = {model: 0 for model in self.student_models}
        
        # Track the distribution of solve rates
        solve_rate_bins = {model: {0.0: 0, '0.01-0.25': 0, '0.26-0.5': 0, '0.51-0.75': 0, '0.76-1.0': 0} 
                          for model in self.student_models}
        
        for problem_idx, hints in self.hint_data.items():
            for hint_text, hint_info in hints.items():
                for model, solve_rate in hint_info["model_performances"].items():
                    hint_total[model] += 1
                    
                    # Categorize solve rates
                    if solve_rate == 0.0:
                        solve_rate_bins[model][0.0] += 1
                    elif solve_rate <= 0.25:
                        solve_rate_bins[model]['0.01-0.25'] += 1
                    elif solve_rate <= 0.5:
                        solve_rate_bins[model]['0.26-0.5'] += 1
                    elif solve_rate <= 0.75:
                        solve_rate_bins[model]['0.51-0.75'] += 1
                    else:
                        solve_rate_bins[model]['0.76-1.0'] += 1
                    
                    if solve_rate > 0:
                        hint_correct[model] += 1
        
        for model in self.student_models:
            if hint_total[model] > 0:
                rate = hint_correct[model] / hint_total[model]
                print(f"  {model}: {hint_correct[model]}/{hint_total[model]} hints with non-zero solve rate ({rate:.4f} or {rate*100:.1f}%)")
                
                # Print solve rate distribution
                print(f"    Solve rate distribution:")
                for bin_name, count in solve_rate_bins[model].items():
                    bin_pct = count / hint_total[model] * 100 if hint_total[model] > 0 else 0
                    print(f"      {bin_name}: {count} hints ({bin_pct:.1f}%)")
        
        # 3. Check for any zero values in the evaluation
        print("\nSample Hint Performance (first 5 problems):")
        for i, (problem_idx, hints) in enumerate(list(self.hint_data.items())[:5]):
            problem_text = self.problem_metadata[problem_idx]["problem"][:50] + "..."
            print(f"\nProblem {problem_idx}: {problem_text}")
            
            for j, (hint_text, hint_info) in enumerate(list(hints.items())[:2]):  # Show first 2 hints per problem
                hint_short = hint_text[:50] + "..." if len(hint_text) > 50 else hint_text
                print(f"  Hint {j+1}: {hint_short}")
                
                for model in self.student_models:
                    if model in hint_info["model_performances"]:
                        solve_rate = hint_info["model_performances"][model]
                        print(f"    {model}: solve_rate = {solve_rate:.4f}")

##############################################################################
# Command-Line Interface
##############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Analysis of individual hint effectiveness.")
    parser.add_argument("--samples", "-s", type=str, required=True,
                        help="Path to the original samples.json (with difficulty bins, etc.).")
    parser.add_argument("--results_dir", "-r", type=str, default="results",
                        help="Directory containing the model results JSON files.")
    parser.add_argument("--output_dir", "-o", type=str, default="hint_analysis",
                        help="Where to save the analysis results.")
    parser.add_argument("--student_models", "-m", type=str, required=True,
                        help="Comma-separated list of student model tags, e.g. 'Llama-3.1-8B-Instruct,gemma-2-2b-it'")
    parser.add_argument("--prompt_names", "-p", type=str, required=True,
                        help="Comma-separated list of prompt names, e.g. 'socratic_question,direct_hint,step_suggestion'")
    return parser.parse_args()

def main():
    args = parse_args()
    student_models = [x.strip() for x in args.student_models.split(",")]
    prompt_names = [x.strip() for x in args.prompt_names.split(",")]
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"[Error] Results directory '{args.results_dir}' does not exist.")
        print("Make sure you've run inference.py with multiple teacher prompts first.")
        sys.exit(1)
    
    # Check if samples file exists
    if not os.path.exists(args.samples):
        print(f"[Error] Samples file '{args.samples}' does not exist.")
        sys.exit(1)
    
    # Print expected file pattern
    print(f"[Hint Analysis] Expecting result files with patterns: ")
    for model in student_models:
        print(f"  - {args.results_dir}/answers_{model}_no_hint.json")
        for prompt in prompt_names:
            print(f"  - {args.results_dir}/answers_{model}_{prompt}_with_hint.json")
    
    # Run the analysis
    analysis = HintAnalysis(
        samples_file=args.samples,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        student_models=student_models,
        prompt_names=prompt_names
    )
    analysis.run()

if __name__ == "__main__":
    main() 