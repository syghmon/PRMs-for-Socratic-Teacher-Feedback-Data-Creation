#!/usr/bin/env python3
"""
analysis.py

Collects the JSON outputs from inference.py for multiple student models (with and without hints),
plus the original samples.json, and generates:

1. Per-problem bar plot of solve rates (for each student model no-hint vs. with-hint, plus LLaMA8B).
2. Bin-based bar plot of average solve rates (0-10% bin, 10-20% bin, etc.).
3. Per-problem difference plots (with-hint minus no-hint).
4. Bin-based difference plots.
5. Difference-of-differences (strongest vs. weakest) per-problem and per-bin.
6. Print out (or save) interesting examples.

Usage:
  python analysis.py --samples data/samples.json \
                     --results_dir results \
                     --output_dir analysis_outputs \
                     --model_tags "ModelA,ModelB,ModelC" \
                     --bins "0-10%,10-20%,20-30%,30-40%,40-50%,50-60%,60-70%,70-80%,80-90%,90-100%"

Where:
  --samples/-s     : Path to original input problems (contains final_answer, difficulty_bin, llama8b_solve_rate).
  --results_dir/-r : Directory where the *_no_hint.json and *_with_hint.json are located.
  --output_dir/-o  : Directory to save the plots and interesting examples.
  --model_tags/-m  : Comma-separated list of model tags (matching what you used in inference).
  --bins/-b        : Comma-separated list of difficulty_bin labels in ascending order.

The script will look for files named:
  {OUTPUT_BASE}_{MODEL_TAG}_no_hint.json
  {OUTPUT_BASE}_{MODEL_TAG}_with_hint.json
For each model tag.

Examples:
  python analysis.py \\
      --samples data/samples.json \\
      --results_dir results \\
      --output_dir analysis_outputs \\
      --model_tags "SmolLM2-1.7B,BigLM13B" \\
      --bins "0-10%,10-20%,20-30%,30-40%,40-50%,50-60%,60-70%,70-80%,80-90%,90-100%"
"""
import argparse
import os
import json
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

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
        data = json.load(f)
    return data

def get_solve_rate(entry):
    """
    Given a result entry from the `_no_hint.json` or `_with_hint.json`,
    return the fraction of correct answers across all samples.
    entry["correctness"] is a list of bools for each sample (if the inference used multiple samples).
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
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

##############################################################################
# Main Analysis Class
##############################################################################

class Analysis:
    """
    Collects data from:
      - The original samples (with fields: problem, answer, difficulty_bin, llama8b_solve_rate, etc.)
      - The multiple JSON result files from inference (for each model, with/without hints).

    Produces:
      - Merged data structure
      - Plots for per-problem and bin-based solve rates
      - Difference plots
      - Interesting examples
    """

    def __init__(self,
                 samples_file: str,
                 results_dir: str,
                 output_dir: str,
                 model_tags: list,
                 bin_labels: list):
        self.samples_file = samples_file
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.model_tags = model_tags
        self.bin_labels = bin_labels

        # Data structure to hold per-problem info
        # all_results[i] = {
        #    "problem": ...
        #    "final_answer": ...
        #    "difficulty_bin": ...
        #    "llama8b_solve_rate": ...
        #    "teacher_hint": ...
        #    "models": {
        #       model_tag: {
        #          "no_hint": <float solve rate>,
        #          "with_hint": <float solve rate>
        #       },
        #       ...
        #    }
        # }
        self.all_results = {}

    def run(self):
        """High-level workflow."""
        self.load_samples()
        self.load_model_results()
        self.plot_per_problem_solve_rates()
        self.plot_bin_based_solve_rates()
        self.plot_per_problem_differences()
        self.plot_bin_based_differences()
        self.plot_difference_of_differences()
        self.save_interesting_examples()
        # Add summary table at the end
        self.create_summary_tables()

    def load_samples(self):
        """Load original samples (problems, final_answer, difficulty_bin, llama8b_solve_rate, etc.)."""
        print(f"[Analysis] Loading samples from {self.samples_file}")
        samples = load_json_data(self.samples_file)
        for i, samp in enumerate(samples):
            self.all_results[i] = {
                "problem": samp["problem"],
                "final_answer": samp.get("answer", ""),
                "difficulty_bin": samp.get("difficulty_bin", None),
                "llama8b_solve_rate": samp.get("llama8b_solve_rate", 0.0),
                "teacher_hint": None,  # to be filled from with-hint files
                "models": {}
            }

    def load_model_results(self):
        """
        For each model tag, read the `_no_hint.json` and `_with_hint.json` files in `results_dir`.
        Then store solve rates and teacher hints (from with-hint).
        """
        for tag in self.model_tags:
            no_hint_file = os.path.join(self.results_dir, f"answers_{tag}_no_hint.json")
            with_hint_file = os.path.join(self.results_dir, f"answers_{tag}_with_hint.json")

            print(f"[Analysis] Loading model results for {tag}")
            no_hint_data = load_json_data(no_hint_file)
            with_hint_data = load_json_data(with_hint_file)

            # Skip this model if both files are missing
            if not no_hint_data and not with_hint_data:
                print(f"[Warning] Skipping model {tag} as no data files were found")
                continue

            # Each data list should have the same length (#problems).
            if no_hint_data and len(no_hint_data) != len(self.all_results):
                print(f"[Warning] Number of problems in {no_hint_file} does not match samples.")
            if with_hint_data and len(with_hint_data) != len(self.all_results):
                print(f"[Warning] Number of problems in {with_hint_file} does not match samples.")

            # Initialize model data for all problems (for this tag)
            for i in range(len(self.all_results)):
                if tag not in self.all_results[i]["models"]:
                    self.all_results[i]["models"][tag] = {
                        "no_hint": 0.0,
                        "with_hint": 0.0
                    }

            # Add no-hint data if available
            for i, entry in enumerate(no_hint_data):
                if i >= len(self.all_results):
                    break
                sr = get_solve_rate(entry)
                self.all_results[i]["models"][tag]["no_hint"] = sr

            # Add with-hint data if available
            for i, entry in enumerate(with_hint_data):
                if i >= len(self.all_results):
                    break
                sr = get_solve_rate(entry)
                self.all_results[i]["models"][tag]["with_hint"] = sr
                # Also store teacher_hint if available
                if "teacher_hint" in entry and entry["teacher_hint"] is not None:
                    self.all_results[i]["teacher_hint"] = entry["teacher_hint"]

    ############################################################################
    # PLOTTING
    ############################################################################

    def plot_per_problem_solve_rates(self):
        """
        Plot a large chart with 7 bars per problem:
          - LLaMA8B solve rate
          - Student1 no hint
          - Student1 with hint
          - Student2 no hint
          - Student2 with hint
          - Student3 no hint
          - Student3 with hint
        This can be extremely wide if there are many problems, so be aware.

        We'll save it as a PNG in output_dir. For large N, consider an alternative
        or an interactive approach. 
        """
        print("[Analysis] Plotting per-problem solve rates (this may be large).")

        num_problems = len(self.all_results)
        # X-axis = problem index
        x_vals = list(range(num_problems))
        # We have 1 bar for LLaMA8B, plus 2 bars per student model
        # => total of 1 + 2*(len(model_tags)) bars.
        # We'll group them with small offsets around each problem index.
        group_size = 1 + 2*len(self.model_tags)
        bar_width = 0.6 / group_size  # 0.6 / group_size to keep them narrower

        # We'll gather the data in a list of lists for each bar category
        # For example:
        #   llama8b_vals = [0.05, 0.1, 0.9, ...] (for each problem)
        #   modelX_no_vals = [...]
        #   modelX_with_vals = [...]
        # We can then call ax.bar multiple times with appropriate offsets.

        # 1) LLaMA8B
        llama8b_vals = [self.all_results[i]["llama8b_solve_rate"] for i in range(num_problems)]

        # We'll create a figure
        fig, ax = plt.subplots(figsize=(min(20, num_problems*0.2), 6))
        # The base offset from each x
        offset = - (group_size * bar_width) / 2.0

        # Plot LLaMA8B
        x_pos_llama = [x + offset for x in x_vals]
        ax.bar(x_pos_llama, llama8b_vals, width=bar_width, label="LLaMA8B")
        offset += bar_width

        # 2) For each model, plot no_hint and with_hint
        for tag in self.model_tags:
            # gather arrays
            no_hint_vals = []
            with_hint_vals = []
            for i in range(num_problems):
                no_hint_vals.append(self.all_results[i]["models"][tag]["no_hint"])
                with_hint_vals.append(self.all_results[i]["models"][tag]["with_hint"])

            # no-hint
            x_pos_no = [x + offset for x in x_vals]
            ax.bar(x_pos_no, no_hint_vals, width=bar_width, label=f"{tag} (no hint)")
            offset += bar_width

            # with-hint
            x_pos_with = [x + offset for x in x_vals]
            ax.bar(x_pos_with, with_hint_vals, width=bar_width, label=f"{tag} (with hint)")
            offset += bar_width

        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(i) for i in x_vals], rotation=90)
        ax.set_xlabel("Problem Index")
        ax.set_ylabel("Solve Rate")
        ax.set_ylim([0, 1])
        ax.set_title("Per-problem Solve Rates (LLaMA8B + Students)")
        ax.legend(ncol=2, fontsize="small")
        fig.tight_layout()

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "per_problem_solve_rates.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  -> Saved per-problem solve rate plot to {out_path}")

    def plot_bin_based_solve_rates(self):
        """
        Group problems by difficulty_bin, compute average solve rates for each group:
          - LLaMA8B
          - Each student model (no_hint, with_hint)
        Then make a grouped bar chart with error bars.
        """
        print("[Analysis] Plotting bin-based solve rates.")
        # Create a map: bin_label -> list of dict
        #   each dict has: {"llama8b": x, "ModelX_no": x, "ModelX_with": x, ...}
        bin_map = defaultdict(list)
        # Also track counts per bin for annotation
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
                item[f"{tag}_no"] = self.all_results[i]["models"][tag]["no_hint"]
                item[f"{tag}_with"] = self.all_results[i]["models"][tag]["with_hint"]
            bin_map[b].append(item)

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
                # No hint stats
                no_vals = [r[f"{tag}_no"] for r in records]
                no_mean, no_stderr = compute_stats(no_vals)
                result[f"{tag}_no_mean"] = no_mean
                result[f"{tag}_no_stderr"] = no_stderr
                
                # With hint stats
                with_vals = [r[f"{tag}_with"] for r in records]
                with_mean, with_stderr = compute_stats(with_vals)
                result[f"{tag}_with_mean"] = with_mean
                result[f"{tag}_with_stderr"] = with_stderr
            
            bin_summary[b] = result

        # Sort bins properly
        all_bins = list(bin_summary.keys())
        bins_in_order = custom_sort_bins(all_bins)
        
        # Print bin statistics for debug
        print("\n[Analysis] Bin Statistics:")
        print(f"{'Bin':<10} {'Count':>6} {'LLaMA8B':>10} {'Average':>10}")
        print("-" * 50)
        for b in bins_in_order:
            print(f"{b:<10} {bin_summary[b]['count']:>6} {bin_summary[b]['llama8b_mean']:>10.2f} {sum(bin_summary[b][f'{tag}_no_mean'] for tag in self.model_tags)/len(self.model_tags):>10.2f}")

        # Apply consistent plot style
        set_plot_style()
        
        # Each bin will have 1 + 2*(len(model_tags)) bars
        group_size = 1 + 2*len(self.model_tags)
        bar_width = 0.6 / group_size
        x_vals = range(len(bins_in_order))
        
        # Create a larger figure for better readability
        fig, ax = plt.subplots(figsize=(max(10, len(bins_in_order)*1.5), 8))
        
        # Use a better color palette
        colors = plt.cm.tab10.colors
        llama_color = "grey"
        
        offset_start = - (group_size * bar_width) / 2.0
        offset = offset_start
        
        # Plot LLaMA8B with error bars
        llama_means = [bin_summary[b]["llama8b_mean"] for b in bins_in_order]
        llama_errors = [bin_summary[b]["llama8b_stderr"] for b in bins_in_order]
        x_pos_llama = [x + offset for x in x_vals]
        ax.bar(x_pos_llama, llama_means, width=bar_width, label="LLaMA8B", 
               color=llama_color, alpha=0.7)
        ax.errorbar(x_pos_llama, llama_means, yerr=llama_errors, fmt='none', 
                   color='black', capsize=3, alpha=0.5)
        offset += bar_width

        # Plot each model pair (no_hint and with_hint)
        for i, tag in enumerate(self.model_tags):
            # Get colors for this model (one for no hint, a darker one for with hint)
            model_color = colors[i % len(colors)]
            model_color_dark = tuple(max(0, c-0.2) for c in model_color[0:3]) + (model_color[3],) if len(model_color) > 3 else model_color
            
            # no-hint data
            no_means = [bin_summary[b][f"{tag}_no_mean"] for b in bins_in_order]
            no_errors = [bin_summary[b][f"{tag}_no_stderr"] for b in bins_in_order]
            x_pos_no = [x + offset for x in x_vals]
            ax.bar(x_pos_no, no_means, width=bar_width, label=f"{tag} (no hint)",
                  color=model_color, alpha=0.7)
            ax.errorbar(x_pos_no, no_means, yerr=no_errors, fmt='none',
                      color='black', capsize=3, alpha=0.5)
            offset += bar_width

            # with-hint data
            with_means = [bin_summary[b][f"{tag}_with_mean"] for b in bins_in_order]
            with_errors = [bin_summary[b][f"{tag}_with_stderr"] for b in bins_in_order]
            x_pos_with = [x + offset for x in x_vals]
            ax.bar(x_pos_with, with_means, width=bar_width, label=f"{tag} (with hint)",
                  color=model_color_dark, alpha=0.7, hatch="//")
            ax.errorbar(x_pos_with, with_means, yerr=with_errors, fmt='none',
                      color='black', capsize=3, alpha=0.5)
            offset += bar_width

        # Add sample counts as annotations
        for i, b in enumerate(bins_in_order):
            count = bin_summary[b]["count"]
            ax.annotate(f"n={count}", xy=(i, 0.02), xytext=(0, 0), 
                      textcoords="offset points", ha='center', va='bottom',
                      color='black', fontweight='bold', fontsize=9)

        # Improve plot appearance
        ax.set_xticks(list(x_vals))
        ax.set_xticklabels(bins_in_order, rotation=45, ha="right")
        ax.set_ylabel("Average Solve Rate")
        ax.set_ylim([0, 1])
        ax.set_xlabel("Difficulty Bin (LLaMA8B solve rate)")
        ax.set_title("Solve Rates by Difficulty Bin", fontweight='bold')
        
        # Add grid lines for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Create a better legend with columns
        legend_cols = min(3, 1 + len(self.model_tags))
        ax.legend(fontsize='small', ncol=legend_cols, loc='upper right', 
                 frameon=True, fancybox=True, framealpha=0.9, 
                 borderpad=1, handlelength=2)
        
        # Add a horizontal line at 0.5 for reference
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        fig.tight_layout()

        # Save the plot
        out_path = os.path.join(self.output_dir, "bin_based_solve_rates.png")
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved bin-based solve rate plot to {out_path}")
        
        # Create a second plot focusing just on the hint gain across bins
        self.plot_bin_based_solve_rates_combined(bin_summary, bins_in_order)
        
    def plot_bin_based_solve_rates_combined(self, bin_summary, bins_in_order):
        """
        Create a combined visualization showing both no-hint and with-hint results
        in a more compact format, focusing on the improvement from hints.
        """
        set_plot_style()
        
        # Create line plot for all models, showing both no-hint and with-hint
        fig, ax = plt.subplots(figsize=(10, 6))
        x_vals = range(len(bins_in_order))
        
        # Plot LLaMA8B baseline as a grey line
        llama_means = [bin_summary[b]["llama8b_mean"] for b in bins_in_order]
        ax.plot(x_vals, llama_means, 'o--', color='grey', linewidth=2, 
                label="LLaMA8B", alpha=0.7, markersize=8)
        
        # Plot each model's no-hint and with-hint results
        colors = plt.cm.tab10.colors
        
        # Plot model results
        for i, tag in enumerate(self.model_tags):
            color = colors[i % len(colors)]
            
            # No hint (solid line)
            no_means = [bin_summary[b][f"{tag}_no_mean"] for b in bins_in_order]
            ax.plot(x_vals, no_means, 'o-', label=f"{tag} (no hint)", 
                    color=color, linewidth=2, markersize=8)
            
            # With hint (dashed line)
            with_means = [bin_summary[b][f"{tag}_with_mean"] for b in bins_in_order]
            ax.plot(x_vals, with_means, 's--', label=f"{tag} (with hint)", 
                    color=color, linewidth=2, markersize=8)
            
            # Fill the area between no-hint and with-hint to highlight improvement
            ax.fill_between(x_vals, no_means, with_means, color=color, alpha=0.2)
        
        # Improve plot appearance
        ax.set_xticks(list(x_vals))
        ax.set_xticklabels(bins_in_order, rotation=45, ha="right")
        ax.set_ylabel("Average Solve Rate")
        ax.set_ylim([0, 1])
        ax.set_xlabel("Difficulty Bin (LLaMA8B solve rate)")
        ax.set_title("Model Performance Comparison With and Without Hints", fontweight='bold')
        
        # Add sample counts as annotations
        for i, b in enumerate(bins_in_order):
            count = bin_summary[b]["count"]
            ax.annotate(f"n={count}", xy=(i, 0.05), xytext=(0, 0), 
                       textcoords="offset points", ha='center', va='bottom',
                       color='black', fontweight='bold', fontsize=9)
        
        # Add grid and reference line
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # Better legend
        ax.legend(fontsize='small', ncol=2, loc='upper right', 
                 frameon=True, fancybox=True)
        
        fig.tight_layout()
        
        # Save the combined plot
        out_path = os.path.join(self.output_dir, "bin_based_solve_rates_combined.png")
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved combined bin-based solve rate plot to {out_path}")

    def plot_per_problem_differences(self):
        """
        Show (with-hint - no-hint) for each problem and each student model.
        Could be a multi-line plot or multiple bar sets.
        We'll do a line plot for clarity.
        """
        print("[Analysis] Plotting per-problem difference (with-hint minus no-hint).")
        num_problems = len(self.all_results)
        x_vals = list(range(num_problems))

        fig, ax = plt.subplots(figsize=(min(20, num_problems*0.2), 6))

        for tag in self.model_tags:
            diffs = []
            for i in range(num_problems):
                no_val = self.all_results[i]["models"][tag]["no_hint"]
                with_val = self.all_results[i]["models"][tag]["with_hint"]
                diffs.append(with_val - no_val)
            ax.plot(x_vals, diffs, label=f"{tag} (with - no)")

        ax.axhline(0, color='gray', linewidth=1)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(i) for i in x_vals], rotation=90)
        ax.set_xlabel("Problem Index")
        ax.set_ylabel("Solve Rate Difference")
        ax.set_title("Per-problem difference: (with-hint) - (no-hint)")
        ax.legend()
        fig.tight_layout()

        out_path = os.path.join(self.output_dir, "per_problem_diff.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  -> Saved per-problem difference plot to {out_path}")

    def plot_bin_based_differences(self):
        """
        For each bin, average (with-hint - no-hint) for each student model.
        Then do a grouped bar chart with error bars. 
        """
        print("[Analysis] Plotting bin-based difference (with-hint minus no-hint).")

        # For each bin, gather differences with full detail for error bars
        bin_data = defaultdict(lambda: defaultdict(list))
        
        # bin_data[ bin_label ][ model_tag ] = list of differences
        for i in range(len(self.all_results)):
            b = self.all_results[i]["difficulty_bin"]
            if b is None:
                continue
            for tag in self.model_tags:
                no_val = self.all_results[i]["models"][tag]["no_hint"]
                with_val = self.all_results[i]["models"][tag]["with_hint"]
                diff = with_val - no_val
                bin_data[b][tag].append(diff)

        # Calculate stats: mean and std error
        bin_summary = {}
        bin_counts = {}
        for b, model_dict in bin_data.items():
            bin_summary[b] = {}
            bin_counts[b] = 0
            
            for tag, diffs in model_dict.items():
                if len(diffs) == 0:
                    bin_summary[b][tag] = {"mean": 0.0, "stderr": 0.0}
                else:
                    mean, stderr = compute_stats(diffs)
                    bin_summary[b][tag] = {"mean": mean, "stderr": stderr}
                    bin_counts[b] = len(diffs)  # All models should have same count per bin

        # Sort bins properly
        all_bins = list(bin_summary.keys())
        bins_in_order = custom_sort_bins(all_bins)
        
        # Print bin statistics for debug
        print("\n[Analysis] Improvement Statistics (with_hint - no_hint):")
        print(f"{'Bin':<10} {'Count':>6} " + " ".join(f"{tag[:10]:>10}" for tag in self.model_tags))
        print("-" * (30 + 10*len(self.model_tags)))
        for b in bins_in_order:
            print(f"{b:<10} {bin_counts[b]:>6} " + " ".join(f"{bin_summary[b][tag]['mean']:>10.2f}" for tag in self.model_tags))

        # Apply plot style
        set_plot_style()
        
        # Create plot
        x_vals = range(len(bins_in_order))
        group_size = len(self.model_tags)
        bar_width = 0.7 / group_size
        fig, ax = plt.subplots(figsize=(max(10, len(bins_in_order)*1.5), 6))
        
        # Use same color scheme as earlier plots for consistency
        colors = plt.cm.tab10.colors
        
        offset_start = - (group_size * bar_width) / 2.0
        offset = offset_start

        # Plot each model's improvement with error bars
        for i, tag in enumerate(self.model_tags):
            model_color = colors[i % len(colors)]
            
            means = [bin_summary[b][tag]["mean"] for b in bins_in_order]
            errors = [bin_summary[b][tag]["stderr"] for b in bins_in_order]
            
            x_pos = [x + offset for x in x_vals]
            ax.bar(x_pos, means, width=bar_width, label=f"{tag}",
                  color=model_color, alpha=0.8)
            ax.errorbar(x_pos, means, yerr=errors, fmt='none',
                      color='black', capsize=3, alpha=0.5)
            
            # Add text labels for the improvement percentage
            for j, (x, mean) in enumerate(zip(x_pos, means)):
                if abs(mean) > 0.05:  # Only label significant improvements
                    ax.annotate(f"{mean:.2f}", 
                              xy=(x, mean + (0.01 if mean >= 0 else -0.05)),
                              ha='center', va='bottom' if mean >= 0 else 'top',
                              fontsize=8)
            
            offset += bar_width

        # Add sample counts as annotations
        for i, b in enumerate(bins_in_order):
            count = bin_counts[b]
            ax.annotate(f"n={count}", xy=(i, 0.01), xytext=(0, -15), 
                      textcoords="offset points", ha='center', va='top',
                      color='black', fontweight='bold', fontsize=9)

        # Improve plot appearance
        ax.axhline(0, color='gray', linewidth=1)
        ax.set_xticks(list(x_vals))
        ax.set_xticklabels(bins_in_order, rotation=45, ha="right")
        ax.set_ylabel("Improvement from Hint (with_hint - no_hint)")
        ax.set_title("Improvement from Teacher Hints by Difficulty Bin", fontweight='bold')
        ax.set_xlabel("Difficulty Bin (LLaMA8B solve rate)")
        
        # Add horizontal line at 0 for reference
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        # Add grid for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Better legend
        ax.legend(fontsize='small', ncol=min(len(self.model_tags), 3), 
                 loc='upper right', frameon=True, fancybox=True)
        
        fig.tight_layout()

        out_path = os.path.join(self.output_dir, "bin_based_diff.png")
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved bin-based difference plot to {out_path}")

    def plot_difference_of_differences(self):
        """
        For each problem, identify which model had the largest difference 
        (with-hint - no-hint) and which had the smallest difference, then
        compute (max_diff - min_diff). We can do a per-problem plot (line),
        and also a bin-based average plot. 
        """
        print("[Analysis] Plotting difference-of-differences (strongest - weakest).")

        set_plot_style()
        
        # 1) Per-problem difference-of-differences
        num_problems = len(self.all_results)
        dvals_per_problem = []
        max_model_per_problem = []
        min_model_per_problem = []
        
        for i in range(num_problems):
            # Track which model had max/min improvement for each problem
            deltas = []
            model_deltas = {}
            
            for tag in self.model_tags:
                no_val = self.all_results[i]["models"][tag]["no_hint"]
                with_val = self.all_results[i]["models"][tag]["with_hint"]
                delta = with_val - no_val
                deltas.append(delta)
                model_deltas[tag] = delta
            
            if deltas:
                maxd = max(deltas)
                mind = min(deltas)
                dvals_per_problem.append(maxd - mind)
                
                # Find which models had max/min improvement
                max_tag = max(model_deltas.items(), key=lambda x: x[1])[0]
                min_tag = min(model_deltas.items(), key=lambda x: x[1])[0]
                max_model_per_problem.append(max_tag)
                min_model_per_problem.append(min_tag)
            else:
                dvals_per_problem.append(0)
                max_model_per_problem.append(None)
                min_model_per_problem.append(None)

        x_vals = list(range(num_problems))
        fig, ax = plt.subplots(figsize=(min(20, num_problems*0.3), 8))
        
        # Plot the difference of differences
        ax.plot(x_vals, dvals_per_problem, 'o-', label="Max - Min improvement", 
                color='purple', linewidth=2, markersize=4, alpha=0.7)
        
        # Highlight points with large differences
        threshold = 0.3  # Highlight points where the difference is >0.3
        highlight_x = [x for x, v in zip(x_vals, dvals_per_problem) if v > threshold]
        highlight_y = [v for v in dvals_per_problem if v > threshold]
        if highlight_x:
            ax.scatter(highlight_x, highlight_y, color='red', s=80, alpha=0.7, 
                       label=f"Large difference (>{threshold})")
        
        # Add annotations for some of the highlighted points
        if highlight_x:
            # Only annotate some of the points to avoid cluttering
            for x, y, max_model, min_model in sorted(zip(highlight_x, highlight_y, 
                                                       [max_model_per_problem[i] for i in highlight_x],
                                                       [min_model_per_problem[i] for i in highlight_x]),
                                                   key=lambda item: item[1], reverse=True)[:min(10, len(highlight_x))]:
                ax.annotate(f"{y:.2f}\n{max_model} vs {min_model}", 
                           xy=(x, y), xytext=(0, 10),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        
        # Add a horizontal line at y=0
        ax.axhline(0, color='gray', linewidth=1, linestyle='--')
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        
        # Improve axis labels and titles
        ax.set_xticks(x_vals[::max(1, len(x_vals)//20)])  # Show fewer x-tick labels
        ax.set_xticklabels([str(i) for i in x_vals[::max(1, len(x_vals)//20)]], rotation=90)
        ax.set_xlabel("Problem Index")
        ax.set_ylabel("Difference-of-differences (max delta - min delta)")
        ax.set_title("Per-problem Gap Between Most and Least Improved Models", fontweight='bold')
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        out_path = os.path.join(self.output_dir, "per_problem_diff_of_diffs.png")
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved difference-of-differences (per-problem) plot to {out_path}")

        try:
            # 2) Bin-based difference-of-differences with model count information
            print("[Analysis] Plotting bin-based difference-of-differences.")
            
            # Ensure we have at least 2 models for meaningful comparison
            if len(self.model_tags) < 2:
                print("[Warning] Need at least 2 models to compute difference-of-differences")
                return
                
            bin_data = defaultdict(list)
            bin_max_models = defaultdict(lambda: defaultdict(int))
            bin_min_models = defaultdict(lambda: defaultdict(int))
            bin_counts = defaultdict(int)
            
            for i in range(num_problems):
                b = self.all_results[i]["difficulty_bin"]
                if b is None:
                    continue
                    
                bin_counts[b] += 1
                
                # Guard against empty model sets or missing data
                if not self.model_tags or not self.all_results[i]["models"]:
                    continue
                
                deltas = []
                model_deltas = {}
                
                for tag in self.model_tags:
                    # Skip if model data is missing
                    if tag not in self.all_results[i]["models"]:
                        continue
                        
                    model_data = self.all_results[i]["models"][tag]
                    if "no_hint" not in model_data or "with_hint" not in model_data:
                        continue
                        
                    no_val = model_data["no_hint"]
                    with_val = model_data["with_hint"]
                    delta = with_val - no_val
                    deltas.append(delta)
                    model_deltas[tag] = delta
                
                # Only proceed if we have model improvements to compare
                if deltas and len(deltas) >= 2:
                    maxd = max(deltas)
                    mind = min(deltas)
                    dval = maxd - mind
                    bin_data[b].append(dval)
                    
                    # Track which models had max/min improvement most often
                    max_tag = max(model_deltas.items(), key=lambda x: x[1])[0]
                    min_tag = min(model_deltas.items(), key=lambda x: x[1])[0]
                    bin_max_models[b][max_tag] += 1
                    bin_min_models[b][min_tag] += 1

            # No bins with data
            if not bin_data:
                print("[Warning] No valid bin data found for difference-of-differences analysis")
                return
                
            # Compute statistics
            bin_summary = {}
            for b, ddiffs in bin_data.items():
                if len(ddiffs) > 0:
                    mean, stderr = compute_stats(ddiffs)
                    bin_summary[b] = {"mean": mean, "stderr": stderr, "count": len(ddiffs)}
                    
                    # Find most common max and min models if they exist
                    if bin_max_models[b]:
                        most_common_max = max(bin_max_models[b].items(), key=lambda x: x[1])[0]
                        bin_summary[b]["max_model"] = most_common_max
                        bin_summary[b]["max_model_count"] = bin_max_models[b][most_common_max]
                    else:
                        bin_summary[b]["max_model"] = None
                        bin_summary[b]["max_model_count"] = 0
                        
                    if bin_min_models[b]:
                        most_common_min = max(bin_min_models[b].items(), key=lambda x: x[1])[0]
                        bin_summary[b]["min_model"] = most_common_min
                        bin_summary[b]["min_model_count"] = bin_min_models[b][most_common_min]
                    else:
                        bin_summary[b]["min_model"] = None
                        bin_summary[b]["min_model_count"] = 0
                else:
                    bin_summary[b] = {"mean": 0.0, "stderr": 0.0, "count": 0,
                                    "max_model": None, "min_model": None,
                                    "max_model_count": 0, "min_model_count": 0}

            # Sort bins properly, handling potential parsing errors
            all_bins = list(bin_summary.keys())
            try:
                bins_in_order = custom_sort_bins(all_bins)
            except Exception as e:
                print(f"[Warning] Error sorting bins: {e}. Using original order.")
                bins_in_order = all_bins
            
            # Print statistics to give more insights
            print("\n[Analysis] Difference-of-differences Statistics (max_delta - min_delta):")
            print(f"{'Bin':<10} {'Count':>6} {'Mean':>8} {'Most Improved':>15} {'Least Improved':>15}")
            print("-" * 60)
            for b in bins_in_order:
                stats = bin_summary[b]
                max_model_info = f"{stats['max_model']} ({stats['max_model_count']})" if stats['max_model'] else "N/A"
                min_model_info = f"{stats['min_model']} ({stats['min_model_count']})" if stats['min_model'] else "N/A"
                print(f"{b:<10} {stats['count']:>6} {stats['mean']:>8.2f} {max_model_info:>15} {min_model_info:>15}")

            x_vals = range(len(bins_in_order))
            y_vals = [bin_summary[b]["mean"] for b in bins_in_order]
            y_errs = [bin_summary[b]["stderr"] for b in bins_in_order]

            fig, ax = plt.subplots(figsize=(max(10, len(bins_in_order)*1.5), 6))
            
            # Plot bars with error bars
            bars = ax.bar(x_vals, y_vals, width=0.6, color="cadetblue", alpha=0.8)
            ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='none', color='black', capsize=4, alpha=0.6)
            
            # Add sample count annotations
            for i, (b, bar) in enumerate(zip(bins_in_order, bars)):
                count = bin_summary[b]["count"]
                ax.annotate(f"n={count}", xy=(i, max(0.01, min(y_vals) * 1.1 if min(y_vals) < 0 else 0.01)), 
                        xytext=(0, -15), 
                        textcoords="offset points", ha='center', va='top',
                        color='black', fontweight='bold', fontsize=9)
                
                # Add annotation for which models had max/min improvement most often
                max_model = bin_summary[b]["max_model"]
                min_model = bin_summary[b]["min_model"]
                if max_model and min_model:
                    # Safe access to bar height
                    bar_height = bar.get_height() if hasattr(bar, 'get_height') else 0
                    
                    label = f"{max_model}\nvs\n{min_model}"
                    ax.annotate(label, xy=(i, bar_height + 0.02), 
                            ha='center', va='bottom', fontsize=8, rotation=0,
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
            
            ax.set_xticks(list(x_vals))
            ax.set_xticklabels(bins_in_order, rotation=45, ha="right")
            ax.set_ylabel("Mean difference between\nmost and least improved models")
            ax.set_title("Model Improvement Gap by Difficulty Bin", fontweight='bold')
            ax.set_xlabel("Difficulty Bin (LLaMA8B solve rate)")
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            
            fig.tight_layout()
            out_path = os.path.join(self.output_dir, "bin_based_diff_of_diffs.png")
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  -> Saved difference-of-differences (bin-based) plot to {out_path}")
            
        except Exception as e:
            print(f"[Error] Failed to generate bin-based difference-of-differences plot: {e}")
            import traceback
            traceback.print_exc()

    ############################################################################
    # INTERESTING EXAMPLES
    ############################################################################

    def save_interesting_examples(self):
        """
        Find problems where at least one model had a large improvement with hint
        or some other interesting condition:
          - e.g. (no_hint==0) -> (with_hint==1)
          - or improvement > 0.5
        Save them to a JSON or CSV in output_dir for manual inspection.
        """
        print("[Analysis] Saving interesting examples.")
        interesting = []

        for i, item in self.all_results.items():
            prob_text = item["problem"]
            teacher_hint = item["teacher_hint"]
            for tag in self.model_tags:
                no_score = item["models"][tag]["no_hint"]
                with_score = item["models"][tag]["with_hint"]
                improvement = with_score - no_score
                # Condition: big jump or from 0 to near 1
                if improvement >= 0.5 or (no_score < 0.1 and with_score > 0.8):
                    example = {
                        "problem_index": i,
                        "problem": prob_text,
                        "teacher_hint": teacher_hint,
                        "model": tag,
                        "score_no_hint": no_score,
                        "score_with_hint": with_score,
                        "improvement": improvement,
                        "difficulty_bin": item["difficulty_bin"],
                        "llama8b_solve_rate": item["llama8b_solve_rate"],
                    }
                    interesting.append(example)

        # Sort descending by improvement
        interesting_sorted = sorted(interesting, key=lambda x: x["improvement"], reverse=True)

        out_path = os.path.join(self.output_dir, "interesting_examples.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(interesting_sorted, f, indent=2)
        print(f"  -> Saved {len(interesting_sorted)} interesting examples to {out_path}")

    def create_summary_tables(self):
        """
        Create comprehensive summary tables with all key metrics in both CSV and text format.
        This makes it easier to analyze the results in a spreadsheet or in the terminal.
        """
        print("[Analysis] Creating summary tables...")
        
        # Group data by bins
        bin_stats = {}
        
        # Count number of problems per bin
        bin_counts = defaultdict(int)
        for i in range(len(self.all_results)):
            b = self.all_results[i]["difficulty_bin"]
            if b is None:
                continue
            bin_counts[b] += 1
        
        # Prepare bins in order
        all_bins = list(bin_counts.keys())
        try:
            bins_in_order = custom_sort_bins(all_bins)
        except Exception as e:
            print(f"[Warning] Error sorting bins: {e}. Using original order.")
            bins_in_order = all_bins
        
        # For each bin, compute:
        # 1. LLaMA8B baseline
        # 2. For each model: no_hint score
        # 3. For each model: with_hint score
        # 4. For each model: improvement
        for b in bins_in_order:
            bin_stats[b] = {
                "count": bin_counts[b],
                "llama8b": 0.0
            }
            
            # Collect problem data for this bin
            problems = []
            for i in range(len(self.all_results)):
                if self.all_results[i]["difficulty_bin"] == b:
                    problems.append(self.all_results[i])
            
            # Compute LLaMA8B average
            llama_sum = sum(p["llama8b_solve_rate"] for p in problems)
            bin_stats[b]["llama8b"] = llama_sum / len(problems) if problems else 0.0
            
            # Compute per-model statistics
            for tag in self.model_tags:
                # No hint score
                no_hint_vals = [p["models"][tag]["no_hint"] for p in problems]
                no_hint_mean, no_hint_stderr = compute_stats(no_hint_vals)
                bin_stats[b][f"{tag}_no_hint"] = no_hint_mean
                bin_stats[b][f"{tag}_no_hint_stderr"] = no_hint_stderr
                
                # With hint score
                with_hint_vals = [p["models"][tag]["with_hint"] for p in problems]
                with_hint_mean, with_hint_stderr = compute_stats(with_hint_vals)
                bin_stats[b][f"{tag}_with_hint"] = with_hint_mean
                bin_stats[b][f"{tag}_with_hint_stderr"] = with_hint_stderr
                
                # Improvement
                improvements = [w - n for w, n in zip(with_hint_vals, no_hint_vals)]
                imp_mean, imp_stderr = compute_stats(improvements)
                bin_stats[b][f"{tag}_improvement"] = imp_mean
                bin_stats[b][f"{tag}_improvement_stderr"] = imp_stderr
        
        # Create CSV file
        csv_file = os.path.join(self.output_dir, "summary_stats.csv")
        with open(csv_file, "w", encoding="utf-8") as f:
            # Write header
            header = ["bin", "count", "llama8b"]
            for tag in self.model_tags:
                header.extend([
                    f"{tag}_no_hint", 
                    f"{tag}_with_hint", 
                    f"{tag}_improvement"
                ])
            f.write(",".join(header) + "\n")
            
            # Write data
            for b in bins_in_order:
                row = [b, str(bin_stats[b]["count"]), f"{bin_stats[b]['llama8b']:.4f}"]
                for tag in self.model_tags:
                    row.extend([
                        f"{bin_stats[b][f'{tag}_no_hint']:.4f}",
                        f"{bin_stats[b][f'{tag}_with_hint']:.4f}",
                        f"{bin_stats[b][f'{tag}_improvement']:.4f}"
                    ])
                f.write(",".join(row) + "\n")
        
        print(f"  -> Saved CSV summary to {csv_file}")
        
        # Create a formatted text table
        txt_file = os.path.join(self.output_dir, "summary_stats.txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            # Title
            f.write("SUMMARY OF TEACHER HINT EFFECTIVENESS\n")
            f.write("=" * 80 + "\n\n")
            
            # Bin-based statistics
            f.write("Performance by Difficulty Bin\n")
            f.write("-" * 80 + "\n")
            
            # Column headers
            header = f"{'Bin':<10} {'Count':>5} {'LLaMA8B':>8}"
            for tag in self.model_tags:
                short_tag = tag.split("-")[0]  # Abbreviate model names if needed
                header += f" {short_tag+'-NoH':>10} {short_tag+'-WithH':>10} {short_tag+'-Imp':>10}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            # Data rows
            for b in bins_in_order:
                row = f"{b:<10} {bin_stats[b]['count']:>5} {bin_stats[b]['llama8b']:.4f}"
                for tag in self.model_tags:
                    row += f" {bin_stats[b][f'{tag}_no_hint']:.4f}    {bin_stats[b][f'{tag}_with_hint']:.4f}    {bin_stats[b][f'{tag}_improvement']:.4f}"
                f.write(row + "\n")
            
            f.write("\n\n")
            
            # Overall statistics (across all bins)
            f.write("Overall Performance (all problems)\n")
            f.write("-" * 50 + "\n")
            
            # Combine all problems
            all_problems = list(self.all_results.values())
            
            # LLaMA8B baseline
            llama_sum = sum(p["llama8b_solve_rate"] for p in all_problems)
            llama_avg = llama_sum / len(all_problems) if all_problems else 0.0
            
            # Model stats
            model_stats = {}
            for tag in self.model_tags:
                # No hint
                no_hint_vals = [p["models"][tag]["no_hint"] for p in all_problems]
                no_hint_mean, no_hint_stderr = compute_stats(no_hint_vals)
                
                # With hint
                with_hint_vals = [p["models"][tag]["with_hint"] for p in all_problems]
                with_hint_mean, with_hint_stderr = compute_stats(with_hint_vals)
                
                # Improvement
                improvements = [w - n for w, n in zip(with_hint_vals, no_hint_vals)]
                imp_mean, imp_stderr = compute_stats(improvements)
                
                model_stats[tag] = {
                    "no_hint": no_hint_mean,
                    "no_hint_stderr": no_hint_stderr,
                    "with_hint": with_hint_mean,
                    "with_hint_stderr": with_hint_stderr,
                    "improvement": imp_mean,
                    "improvement_stderr": imp_stderr
                }
            
            # Write overall section
            f.write(f"Total problems: {len(all_problems)}\n")
            f.write(f"LLaMA8B average: {llama_avg:.4f}\n\n")
            
            f.write(f"{'Model':<20} {'No Hint':>10} {'With Hint':>12} {'Improvement':>14}\n")
            f.write("-" * 60 + "\n")
            
            for tag in self.model_tags:
                stats = model_stats[tag]
                f.write(f"{tag:<20} {stats['no_hint']:.4f} ±{stats['no_hint_stderr']:.4f}  "
                        f"{stats['with_hint']:.4f} ±{stats['with_hint_stderr']:.4f}  "
                        f"{stats['improvement']:.4f} ±{stats['improvement_stderr']:.4f}\n")
            
            # Add Relative Improvement
            f.write("\nRelative Improvement Compared to LLaMA8B Baseline\n")
            f.write("-" * 60 + "\n")
            
            for tag in self.model_tags:
                rel_no_hint = model_stats[tag]["no_hint"] / llama_avg if llama_avg else 0
                rel_with_hint = model_stats[tag]["with_hint"] / llama_avg if llama_avg else 0
                rel_improvement = rel_with_hint - rel_no_hint
                
                f.write(f"{tag:<20} {rel_no_hint:.2%}  →  {rel_with_hint:.2%}  "
                        f"({rel_improvement:+.2%})\n")
        
        print(f"  -> Saved text summary to {txt_file}")


##############################################################################
# Command-Line Interface
##############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Analysis of teacher-student inference results.")
    parser.add_argument("--samples", "-s", type=str, required=True,
                        help="Path to the original samples.json (with difficulty bins, etc.).")
    parser.add_argument("--results_dir", "-r", type=str, default="results",
                        help="Directory containing the model results JSON files.")
    parser.add_argument("--output_dir", "-o", type=str, default="analysis_outputs",
                        help="Where to save the plots and interesting examples.")
    parser.add_argument("--model_tags", "-m", type=str, required=True,
                        help="Comma-separated list of model tags, e.g. 'SmolLM,BigLM'")
    parser.add_argument("--bins", "-b", type=str, default="0-10%,10-20%,20-30%,30-40%,40-50%,50-60%,60-70%,70-80%,80-90%,90-100%",
                        help="Comma-separated list of difficulty bin labels in ascending order.")
    return parser.parse_args()

def main():
    args = parse_args()
    model_tags = [x.strip() for x in args.model_tags.split(",")]
    bin_labels = [x.strip() for x in args.bins.split(",")]

    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"[Warning] Results directory '{args.results_dir}' does not exist.")
        print("Make sure you've run inference.py first to generate result files.")
        if not input("Continue anyway? (y/n): ").lower().startswith('y'):
            sys.exit(1)
    
    # Print expected file pattern
    print(f"[Analysis] Expecting result files with pattern: ")
    for tag in model_tags:
        print(f"  - {args.results_dir}/answers_{tag}_no_hint.json")
        print(f"  - {args.results_dir}/answers_{tag}_with_hint.json")
    
    os.makedirs(args.output_dir, exist_ok=True)

    analysis = Analysis(
        samples_file=args.samples,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        model_tags=model_tags,
        bin_labels=bin_labels
    )
    analysis.run()
    print("[Analysis] Done.")

if __name__ == "__main__":
    main() 