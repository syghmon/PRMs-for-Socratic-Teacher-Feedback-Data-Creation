#!/usr/bin/env python3
"""
analysis.py

Collects the JSON outputs from inference.py for multiple student models (with and without hints),
plus the original samples.json, and generates:

1. Bin-based combined solve rates (showing improvement from hints for different models)
2. Interesting examples (problems where models showed significant improvement with hints)

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
  --bins/-b        : Comma-separated list of difficulty bin labels in ascending order.
"""
import argparse
import os
import json
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings

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
# Main Analysis Class
##############################################################################

class Analysis:
    """
    Collects data from:
      - The original samples (with fields: problem, answer, difficulty_bin, llama8b_solve_rate, etc.)
      - The multiple JSON result files from inference (for each model, with/without hints).

    Produces:
      - Bin-based solve rates combined plot
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
        print(f"[Analysis] Creating output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
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
            
        self.plot_bin_based_solve_rates()
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
                "teacher_hint": None,  # to be filled from with-hint files
                "models": {}
            }

    def load_model_results(self):
        """
        For each model tag, read the `_no_hint.json` and `_with_hint.json` files in `results_dir`.
        Then store solve rates and teacher hints (from with-hint).
        """
        models_loaded = 0
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
                
            models_loaded += 1

            # Each data list should have the same length (#problems).
            if no_hint_data and len(no_hint_data) != len(self.all_results):
                print(f"[Warning] Number of problems in {no_hint_file} ({len(no_hint_data)}) does not match samples ({len(self.all_results)}).")
            if with_hint_data and len(with_hint_data) != len(self.all_results):
                print(f"[Warning] Number of problems in {with_hint_file} ({len(with_hint_data)}) does not match samples ({len(self.all_results)}).")

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
                    
        print(f"[Analysis] Loaded data for {models_loaded} models")

    def plot_bin_based_solve_rates(self):
        """
        Group problems by difficulty_bin, compute average solve rates for each group,
        and create a combined plot showing both no-hint and with-hint results.
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
                if tag in self.all_results[i]["models"]:
                    item[f"{tag}_no"] = self.all_results[i]["models"][tag]["no_hint"]
                    item[f"{tag}_with"] = self.all_results[i]["models"][tag]["with_hint"]
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
                
                # With hint stats
                with_vals = [r[f"{tag}_with"] for r in records if f"{tag}_with" in r]
                with_mean, with_stderr = compute_stats(with_vals)
                result[f"{tag}_with_mean"] = with_mean
                result[f"{tag}_with_stderr"] = with_stderr
            
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
        print(f"{'Bin':<10} {'Count':>6} {'LLaMA8B':>10} {'Average':>10}")
        print("-" * 50)
        for b in bins_in_order:
            model_means = []
            for tag in self.model_tags:
                if f"{tag}_no_mean" in bin_summary[b]:
                    model_means.append(bin_summary[b][f"{tag}_no_mean"])
            
            avg_no_hint = sum(model_means) / len(model_means) if model_means else 0.0
            print(f"{b:<10} {bin_summary[b]['count']:>6} {bin_summary[b]['llama8b_mean']:>10.2f} {avg_no_hint:>10.2f}")
        
        # Create a combined visualization showing both no-hint and with-hint results
        self.plot_bin_based_solve_rates_combined(bin_summary, bins_in_order)
        
    def plot_bin_based_solve_rates_combined(self, bin_summary, bins_in_order):
        """
        Create a combined visualization showing both no-hint and with-hint results
        in a more compact format, focusing on the improvement from hints.
        """
        try:
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
                
                # Check if we have data for this model
                if not all(f"{tag}_no_mean" in bin_summary[b] for b in bins_in_order):
                    print(f"[Warning] Skipping model {tag} in plot due to missing data")
                    continue
                
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
        except Exception as e:
            print(f"[Error] Failed to generate bin-based solve rates plot: {e}")
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
            teacher_hint = item["teacher_hint"]
            for tag in self.model_tags:
                if tag not in item["models"]:
                    continue
                    
                no_score = item["models"][tag]["no_hint"]
                with_score = item["models"][tag]["with_hint"]
                improvement = with_score - no_score
                # Condition: big jump or from 0 to near 1
                if improvement >= 0.5 or (no_score < 0.1 and with_score > 0.8) or improvement <= -0.2:
                    example_type = "improvement" if improvement >= 0 else "regression"
                    example = {
                        "problem_index": i,
                        "problem": prob_text,
                        "teacher_hint": teacher_hint,
                        "model": tag,
                        "score_no_hint": no_score,
                        "score_with_hint": with_score,
                        "improvement": improvement,
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
    
    # Check if samples file exists
    if not os.path.exists(args.samples):
        print(f"[Error] Samples file '{args.samples}' does not exist.")
        sys.exit(1)
    
    # Print expected file pattern
    print(f"[Analysis] Expecting result files with pattern: ")
    for tag in model_tags:
        print(f"  - {args.results_dir}/answers_{tag}_no_hint.json")
        print(f"  - {args.results_dir}/answers_{tag}_with_hint.json")

    analysis = Analysis(
        samples_file=args.samples,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        model_tags=model_tags,
        bin_labels=bin_labels
    )
    analysis.run()

if __name__ == "__main__":
    main() 