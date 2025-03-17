#!/usr/bin/env python3
"""
prompt_analysis.py

Analyzes the effectiveness of different teacher prompts by comparing student performance
with hints generated from these prompts.

Usage:
  python prompt_analysis.py --samples data/samples.json \
                           --results_dir results \
                           --output_dir prompt_analysis \
                           --student_models "Llama-3.1-8B-Instruct,gemma-2-2b-it,gemma-2-9b-it" \
                           --prompt_names "socratic_question,direct_hint,step_suggestion" \
                           --bins "0-10%,10-20%,20-30%,30-40%,40-50%,50-60%,60-70%,70-80%,80-90%,90-100%"

Where:
  --samples/-s       : Path to original input problems (contains final_answer, difficulty_bin, llama8b_solve_rate).
  --results_dir/-r   : Directory where the result files are located (from inference.py).
  --output_dir/-o    : Directory to save the analysis results.
  --student_models/-m: Comma-separated list of student model tags.
  --prompt_names/-p  : Comma-separated list of teacher prompt names (from config.json).
  --bins/-b          : Comma-separated list of difficulty bin labels in ascending order.
"""
import argparse
import os
import json
import sys
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
import numpy as np
from typing import List, Optional, Dict, Tuple

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

##############################################################################
# Main Prompt Analysis Class
##############################################################################

class PromptAnalysis:
    """
    Collects data from multiple teacher prompt runs and analyzes how different prompts
    affect student model performance.
    
    The key difference from the standard analysis is that we focus on comparing prompts
    for each student model rather than comparing students.
    """

    def __init__(self,
                 samples_file: str,
                 results_dir: str,
                 output_dir: str,
                 student_models: list,
                 prompt_names: list,
                 bin_labels: list):
        self.samples_file = samples_file
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.student_models = student_models
        self.prompt_names = prompt_names
        self.bin_labels = bin_labels

        # Data structure to hold per-problem info grouped by student model
        # structured as:
        # {
        #   student_model: {
        #     "no_hint": {
        #       problem_idx: solve_rate,
        #       ...
        #     },
        #     prompt_name1: {
        #       problem_idx: {
        #         "solve_rate": float,
        #         "hint_text": str,
        #       },
        #       ...
        #     },
        #     prompt_name2: {...},
        #     ...
        #   },
        #   ...
        # }
        self.model_data = {model: {"no_hint": {}} for model in student_models}
        
        # Problem metadata stored separately
        # {
        #   problem_idx: {
        #     "problem": str,
        #     "final_answer": str,
        #     "difficulty_bin": str,
        #     "llama8b_solve_rate": float,
        #   },
        #   ...
        # }
        self.problem_metadata = {}

    def run(self):
        """High-level workflow for prompt analysis."""
        print(f"[Prompt Analysis] Creating output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.load_samples()
        if not self.problem_metadata:
            print("[Error] No samples loaded. Exiting.")
            return
            
        self.load_results()
        if not all(bool(self.model_data[model]["no_hint"]) for model in self.student_models):
            print("[Error] Missing no-hint data for some models. Exiting.")
            return
            
        for model in self.student_models:
            missing_prompts = [p for p in self.prompt_names if p not in self.model_data[model]]
            if missing_prompts:
                print(f"[Warning] Missing data for model {model} with prompts: {', '.join(missing_prompts)}")
        
        # Run analysis for each student model
        for model in self.student_models:
            print(f"\n[Prompt Analysis] Analyzing model: {model}")
            self.plot_prompt_comparison(model)
            self.plot_bin_based_comparison(model)
            
        self.save_interesting_examples()
        print("[Prompt Analysis] Done.")

    def load_samples(self):
        """Load original samples and their metadata."""
        print(f"[Prompt Analysis] Loading samples from {self.samples_file}")
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
        We need:
        1. No-hint results for each student model
        2. With-hint results for each (student model, prompt) combination
        """
        print(f"[Prompt Analysis] Loading results from {self.results_dir}")
        
        # First, load no-hint results for each student model
        for model in self.student_models:
            no_hint_file = os.path.join(self.results_dir, f"answers_{model}_no_hint.json")
            no_hint_data = load_json_data(no_hint_file)
            
            if not no_hint_data:
                print(f"[Warning] No no-hint data found for {model}")
                continue
                
            # Extract solve rates
            for i, entry in enumerate(no_hint_data):
                if i >= len(self.problem_metadata):
                    break
                self.model_data[model]["no_hint"][i] = get_solve_rate(entry)
        
        # Then, load with-hint results for each (student model, prompt) combination
        for model in self.student_models:
            for prompt in self.prompt_names:
                with_hint_file = os.path.join(self.results_dir, f"answers_{model}_{prompt}_with_hint.json")
                with_hint_data = load_json_data(with_hint_file)
                
                if not with_hint_data:
                    print(f"[Warning] No with-hint data found for {model} with prompt {prompt}")
                    continue
                
                # Initialize prompt data for this model
                if prompt not in self.model_data[model]:
                    self.model_data[model][prompt] = {}
                
                # Extract solve rates and hints
                for i, entry in enumerate(with_hint_data):
                    if i >= len(self.problem_metadata):
                        break
                    
                    hint = entry.get("teacher_hint", "")
                    solve_rate = get_solve_rate(entry)
                    
                    self.model_data[model][prompt][i] = {
                        "solve_rate": solve_rate,
                        "hint_text": hint
                    }
        
        # Print summary of loaded data
        print("\n[Prompt Analysis] Data Loading Summary:")
        print(f"{'Model':<20} {'No Hint':^10} {' '.join([f'{p:^15}' for p in self.prompt_names])}")
        print("-" * (20 + 10 + 15 * len(self.prompt_names)))
        
        for model in self.student_models:
            no_hint_count = len(self.model_data[model]["no_hint"])
            prompt_counts = []
            
            for prompt in self.prompt_names:
                if prompt in self.model_data[model]:
                    prompt_counts.append(len(self.model_data[model][prompt]))
                else:
                    prompt_counts.append(0)
            
            print(f"{model:<20} {no_hint_count:^10} {' '.join([f'{count:^15}' for count in prompt_counts])}")

    def plot_prompt_comparison(self, model):
        """
        Create visualizations comparing the effectiveness of different prompts 
        for a specific student model.
        """
        print(f"[Prompt Analysis] Plotting prompt comparison for {model}")
        
        if "no_hint" not in self.model_data[model] or not self.model_data[model]["no_hint"]:
            print(f"[Warning] No no-hint data available for {model}. Skipping prompt comparison.")
            return
            
        # Collect data for available prompts
        available_prompts = [p for p in self.prompt_names if p in self.model_data[model] and self.model_data[model][p]]
        if not available_prompts:
            print(f"[Warning] No prompt data available for {model}. Skipping prompt comparison.")
            return
        
        # Calculate overall performance for each prompt
        performance_data = {"no_hint": []}
        for prompt in available_prompts:
            performance_data[prompt] = []
        
        for i in range(len(self.problem_metadata)):
            # Check if we have no-hint data for this problem
            if i in self.model_data[model]["no_hint"]:
                performance_data["no_hint"].append(self.model_data[model]["no_hint"][i])
            
            # Check if we have with-hint data for each prompt for this problem
            for prompt in available_prompts:
                if i in self.model_data[model][prompt]:
                    performance_data[prompt].append(self.model_data[model][prompt][i]["solve_rate"])
        
        # Calculate mean and std error for each prompt
        prompt_stats = {}
        for prompt_type, values in performance_data.items():
            if values:
                mean, stderr = compute_stats(values)
                prompt_stats[prompt_type] = {"mean": mean, "stderr": stderr}
        
        # Sort prompts by performance for better visualization
        sorted_prompts = sorted(
            prompt_stats.keys(),
            key=lambda x: prompt_stats[x]["mean"],
            reverse=True
        )
        
        # Create bar plot comparing prompt effectiveness
        try:
            set_plot_style()
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x_pos = np.arange(len(sorted_prompts))
            means = [prompt_stats[p]["mean"] for p in sorted_prompts]
            stderrs = [prompt_stats[p]["stderr"] for p in sorted_prompts]
            
            # Use different colors for no-hint vs prompts
            colors = ['lightgrey' if p == 'no_hint' else plt.cm.tab10.colors[i % len(plt.cm.tab10.colors)] 
                     for i, p in enumerate(sorted_prompts)]
            
            # Plot bars
            bars = ax.bar(x_pos, means, yerr=stderrs, capsize=5, color=colors, alpha=0.7)
            
            # Add value labels on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                label_text = f"{means[i]:.3f}"
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       label_text, ha='center', va='bottom', fontsize=9)
            
            # Add improvement over no-hint
            if "no_hint" in prompt_stats:
                no_hint_mean = prompt_stats["no_hint"]["mean"]
                for i, prompt in enumerate(sorted_prompts):
                    if prompt != "no_hint":
                        improvement = prompt_stats[prompt]["mean"] - no_hint_mean
                        color = 'green' if improvement > 0 else 'red'
                        ax.text(i, means[i] + 0.03,
                               f"{improvement:+.3f}", ha='center', fontweight='bold', color=color)
            
            # Customize plot
            ax.set_xticks(x_pos)
            ax.set_xticklabels([p.replace('_', ' ').title() if p != 'no_hint' else 'No Hint' for p in sorted_prompts],
                              rotation=25, ha='right')
            ax.set_ylabel('Average Solve Rate')
            ax.set_title(f'Prompt Effectiveness Comparison - {model}', fontweight='bold')
            ax.set_ylim(0, min(1.0, max(means) * 1.2))
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add horizontal line at no-hint performance level
            if "no_hint" in prompt_stats:
                ax.axhline(y=no_hint_mean, color='red', linestyle='--', alpha=0.5)
            
            # Add sample size annotation
            ax.annotate(f"n={len(performance_data['no_hint'])} problems",
                       xy=(0.5, 0.01), xycoords='axes fraction',
                       ha='center', va='bottom',
                       fontsize=9, style='italic')
            
            plt.tight_layout()
            
            # Save the plot
            out_path = os.path.join(self.output_dir, f"{model}_prompt_comparison.png")
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  -> Saved prompt comparison plot to {out_path}")
            
            # Save performance stats as JSON
            stats_path = os.path.join(self.output_dir, f"{model}_prompt_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                stats_data = {
                    "model": model,
                    "performance": {
                        p: {
                            "mean_solve_rate": prompt_stats[p]["mean"],
                            "stderr": prompt_stats[p]["stderr"],
                            "improvement_over_no_hint": (
                                prompt_stats[p]["mean"] - prompt_stats["no_hint"]["mean"] 
                                if p != "no_hint" and "no_hint" in prompt_stats else None
                            )
                        } for p in sorted_prompts
                    },
                    "sample_size": len(performance_data["no_hint"])
                }
                json.dump(stats_data, f, indent=2)
            print(f"  -> Saved performance stats to {stats_path}")
            
        except Exception as e:
            print(f"[Error] Failed to create prompt comparison plot for {model}: {e}")
            import traceback
            traceback.print_exc()

    def plot_bin_based_comparison(self, model):
        """
        Create bin-based visualizations comparing the effectiveness of different prompts
        for a specific student model across difficulty bins.
        """
        print(f"[Prompt Analysis] Plotting bin-based prompt comparison for {model}")
        
        if "no_hint" not in self.model_data[model] or not self.model_data[model]["no_hint"]:
            print(f"[Warning] No no-hint data available for {model}. Skipping bin-based comparison.")
            return
            
        # Collect data for available prompts
        available_prompts = [p for p in self.prompt_names if p in self.model_data[model] and self.model_data[model][p]]
        if not available_prompts:
            print(f"[Warning] No prompt data available for {model}. Skipping bin-based comparison.")
            return
        
        # Group performance data by difficulty bin
        bin_data = defaultdict(lambda: {"no_hint": [], "prompts": {p: [] for p in available_prompts}})
        bin_counts = defaultdict(int)
        
        for i, metadata in self.problem_metadata.items():
            bin_label = metadata.get("difficulty_bin")
            if not bin_label:
                continue
                
            bin_counts[bin_label] += 1
            
            # Add no-hint performance if available
            if i in self.model_data[model]["no_hint"]:
                bin_data[bin_label]["no_hint"].append(self.model_data[model]["no_hint"][i])
            
            # Add with-hint performance for each prompt if available
            for prompt in available_prompts:
                if i in self.model_data[model][prompt]:
                    bin_data[bin_label]["prompts"][prompt].append(
                        self.model_data[model][prompt][i]["solve_rate"]
                    )
        
        if not bin_data:
            print(f"[Warning] No bin data available for {model}. Skipping bin-based comparison.")
            return
        
        # Calculate stats for each bin and prompt
        bin_stats = {}
        for bin_label, data in bin_data.items():
            bin_stats[bin_label] = {
                "count": bin_counts[bin_label],
                "no_hint": compute_stats(data["no_hint"]),
            }
            
            for prompt in available_prompts:
                bin_stats[bin_label][prompt] = compute_stats(data["prompts"][prompt])
        
        # Sort bins
        if self.bin_labels and all(b in bin_stats for b in self.bin_labels):
            sorted_bins = self.bin_labels
        else:
            try:
                sorted_bins = custom_sort_bins(list(bin_stats.keys()))
            except Exception as e:
                print(f"[Warning] Error sorting bins: {e}. Using original order.")
                sorted_bins = list(bin_stats.keys())
        
        # Create line plot
        try:
            set_plot_style()
            fig, ax = plt.subplots(figsize=(12, 7))
            
            x = range(len(sorted_bins))
            
            # Plot no-hint performance
            no_hint_means = [bin_stats[b]["no_hint"][0] for b in sorted_bins]
            no_hint_stderrs = [bin_stats[b]["no_hint"][1] for b in sorted_bins]
            
            ax.plot(x, no_hint_means, 'o-', label="No Hint", 
                   color='black', linewidth=2, markersize=8, alpha=0.7)
            
            # Plot each prompt's performance
            colors = plt.cm.tab10.colors
            for i, prompt in enumerate(available_prompts):
                color = colors[i % len(colors)]
                
                prompt_means = [bin_stats[b][prompt][0] for b in sorted_bins]
                prompt_stderrs = [bin_stats[b][prompt][1] for b in sorted_bins]
                
                ax.plot(x, prompt_means, 'o-', label=prompt.replace('_', ' ').title(), 
                       color=color, linewidth=2, markersize=8)
                
                # Fill the area between no-hint and with-hint to highlight improvement
                ax.fill_between(x, no_hint_means, prompt_means, color=color, alpha=0.1)
            
            # Customize plot
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_bins, rotation=45, ha="right")
            ax.set_ylabel("Average Solve Rate")
            ax.set_ylim([0, 1])
            ax.set_xlabel("Difficulty Bin (LLaMA8B solve rate)")
            ax.set_title(f"Prompt Effectiveness by Difficulty Bin - {model}", fontweight='bold')
            
            # Add sample counts as annotations
            for i, b in enumerate(sorted_bins):
                count = bin_stats[b]["count"]
                ax.annotate(f"n={count}", xy=(i, 0.05), xytext=(0, 0), 
                           textcoords="offset points", ha='center', va='bottom',
                           color='black', fontweight='bold', fontsize=9)
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize='small', ncol=2, loc='upper right', 
                     frameon=True, fancybox=True)
            
            plt.tight_layout()
            
            # Save the plot
            out_path = os.path.join(self.output_dir, f"{model}_bin_based_comparison.png")
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  -> Saved bin-based comparison plot to {out_path}")
            
            # Save bin stats as JSON
            stats_path = os.path.join(self.output_dir, f"{model}_bin_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                bin_data = {
                    "model": model,
                    "bin_performance": {
                        b: {
                            "count": bin_stats[b]["count"],
                            "no_hint": {
                                "mean": bin_stats[b]["no_hint"][0],
                                "stderr": bin_stats[b]["no_hint"][1]
                            },
                            "prompts": {
                                p: {
                                    "mean": bin_stats[b][p][0],
                                    "stderr": bin_stats[b][p][1],
                                    "improvement": bin_stats[b][p][0] - bin_stats[b]["no_hint"][0]
                                } for p in available_prompts
                            }
                        } for b in sorted_bins
                    }
                }
                json.dump(bin_data, f, indent=2)
            print(f"  -> Saved bin stats to {stats_path}")
            
        except Exception as e:
            print(f"[Error] Failed to create bin-based comparison plot for {model}: {e}")
            import traceback
            traceback.print_exc()

    def save_interesting_examples(self):
        """
        Find interesting examples where different prompts had significantly different effects
        on student performance.
        """
        print("[Prompt Analysis] Finding interesting examples...")
        interesting_examples = []
        
        for i, metadata in self.problem_metadata.items():
            for model in self.student_models:
                if "no_hint" not in self.model_data[model] or i not in self.model_data[model]["no_hint"]:
                    continue
                    
                no_hint_rate = self.model_data[model]["no_hint"][i]
                
                # Collect performance for all available prompts
                prompt_performances = {}
                for prompt in self.prompt_names:
                    if prompt in self.model_data[model] and i in self.model_data[model][prompt]:
                        prompt_performances[prompt] = {
                            "solve_rate": self.model_data[model][prompt][i]["solve_rate"],
                            "hint_text": self.model_data[model][prompt][i]["hint_text"]
                        }
                
                if len(prompt_performances) < 2:
                    continue  # Need at least 2 prompts to compare
                
                # Find the best and worst performing prompts
                best_prompt = max(prompt_performances.items(), key=lambda x: x[1]["solve_rate"])
                worst_prompt = min(prompt_performances.items(), key=lambda x: x[1]["solve_rate"])
                
                # Calculate performance range (difference between best and worst)
                performance_range = best_prompt[1]["solve_rate"] - worst_prompt[1]["solve_rate"]
                
                # If the performance range is significant (>0.5) or
                # if one prompt does much better than no-hint (>0.5) while another doesn't (<0.2)
                if (performance_range > 0.5 or 
                    (best_prompt[1]["solve_rate"] - no_hint_rate > 0.5 and 
                     worst_prompt[1]["solve_rate"] - no_hint_rate < 0.2)):
                    
                    example = {
                        "problem_index": i,
                        "problem": metadata["problem"],
                        "final_answer": metadata["final_answer"],
                        "difficulty_bin": metadata["difficulty_bin"],
                        "llama8b_solve_rate": metadata["llama8b_solve_rate"],
                        "student_model": model,
                        "no_hint_solve_rate": no_hint_rate,
                        "prompts": {
                            p: {
                                "hint": data["hint_text"],
                                "solve_rate": data["solve_rate"],
                                "improvement": data["solve_rate"] - no_hint_rate
                            } for p, data in prompt_performances.items()
                        },
                        "best_prompt": best_prompt[0],
                        "worst_prompt": worst_prompt[0],
                        "performance_range": performance_range
                    }
                    interesting_examples.append(example)
        
        if not interesting_examples:
            print("[Warning] No interesting examples found.")
            return
            
        # Sort by performance range (descending)
        interesting_examples.sort(key=lambda x: x["performance_range"], reverse=True)
        
        # Save to file
        out_path = os.path.join(self.output_dir, "interesting_examples.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(interesting_examples, f, indent=2)
            
        print(f"  -> Saved {len(interesting_examples)} interesting examples to {out_path}")

##############################################################################
# Command-Line Interface
##############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Analysis of teacher prompt effectiveness.")
    parser.add_argument("--samples", "-s", type=str, required=True,
                        help="Path to the original samples.json (with difficulty bins, etc.).")
    parser.add_argument("--results_dir", "-r", type=str, default="results",
                        help="Directory containing the model results JSON files.")
    parser.add_argument("--output_dir", "-o", type=str, default="prompt_analysis",
                        help="Where to save the analysis results.")
    parser.add_argument("--student_models", "-m", type=str, required=True,
                        help="Comma-separated list of student model tags, e.g. 'Llama-3.1-8B-Instruct,gemma-2-2b-it'")
    parser.add_argument("--prompt_names", "-p", type=str, required=True,
                        help="Comma-separated list of prompt names, e.g. 'socratic_question,direct_hint,step_suggestion'")
    parser.add_argument("--bins", "-b", type=str, default="0-10%,10-20%,20-30%,30-40%,40-50%,50-60%,60-70%,70-80%,80-90%,90-100%",
                        help="Comma-separated list of difficulty bin labels in ascending order.")
    return parser.parse_args()

def main():
    args = parse_args()
    student_models = [x.strip() for x in args.student_models.split(",")]
    prompt_names = [x.strip() for x in args.prompt_names.split(",")]
    bin_labels = [x.strip() for x in args.bins.split(",")]
    
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
    print(f"[Prompt Analysis] Expecting result files with patterns: ")
    for model in student_models:
        print(f"  - {args.results_dir}/answers_{model}_no_hint.json")
        for prompt in prompt_names:
            print(f"  - {args.results_dir}/answers_{model}_{prompt}_with_hint.json")
    
    # Run the analysis
    analysis = PromptAnalysis(
        samples_file=args.samples,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        student_models=student_models,
        prompt_names=prompt_names,
        bin_labels=bin_labels
    )
    analysis.run()

if __name__ == "__main__":
    main() 