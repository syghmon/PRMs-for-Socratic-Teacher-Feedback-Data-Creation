import json
import os
import pathlib
import random
import sys
from collections import defaultdict

# Get the project root directory (parent of scripts)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()

def sample_joined_problems(
    input_file="data/joined_math_datasets.json", 
    output_file="data/joined_samples.json", 
    samples_per_bin=10, 
    seed=None
):
    """
    Sample problems from the joined BigMath-NuminaMath dataset, ensuring an equal 
    distribution across difficulty bins.
    
    Args:
        input_file: Path to the joined dataset file
        output_file: Path to save the sampled problems
        samples_per_bin: Number of samples to select from each difficulty bin
        seed: Random seed for reproducibility
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Ensure input_file is an absolute path
    if not os.path.isabs(input_file):
        input_file = os.path.join(PROJECT_ROOT, input_file)
    
    # Ensure output_file is an absolute path
    if not os.path.isabs(output_file):
        output_file = os.path.join(PROJECT_ROOT, output_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load the joined dataset
    print(f"Loading joined dataset from {input_file}")
    try:
        with open(input_file, "r") as f:
            joined_data = json.load(f)
        print(f"Loaded {len(joined_data)} problems from joined dataset")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File {input_file} is not a valid JSON file")
        sys.exit(1)
    
    # Filter out any problematic entries
    filtered_data = []
    for item in joined_data:
        # Check required fields
        if (item.get("problem") and 
            item.get("answer") and 
            item.get("solution") and 
            item.get("llama8b_solve_rate") is not None and
            item.get("difficulty_bin")):
            
            # Fix difficulty bin format (remove % if present)
            difficulty_bin = item["difficulty_bin"]
            if isinstance(difficulty_bin, str) and difficulty_bin.endswith("%"):
                item["difficulty_bin"] = difficulty_bin.rstrip("%")
            
            filtered_data.append(item)
    
    print(f"Found {len(filtered_data)} valid problems after filtering")
    
    # Group problems by difficulty bin
    bins = defaultdict(list)
    for item in filtered_data:
        bins[item["difficulty_bin"]].append(item)
    
    # Report bin statistics
    print("\nDifficulty bin distribution:")
    for bin_name, problems in sorted(bins.items()):
        print(f"  {bin_name}: {len(problems)} problems")
    
    # Sample problems from each bin
    sampled_problems = []
    for bin_name, problems in sorted(bins.items()):
        bin_count = min(samples_per_bin, len(problems))
        if bin_count < samples_per_bin:
            print(f"Warning: Only {bin_count} problems available in bin {bin_name}")
        
        bin_samples = random.sample(problems, bin_count)
        sampled_problems.extend(bin_samples)
    
    # Sort by difficulty bin for easier inspection
    sampled_problems.sort(key=lambda x: x["difficulty_bin"])
    
    # Format problems for compatibility with inference.py
    formatted_problems = []
    for item in sampled_problems:
        formatted_item = {
            "problem": item["problem"],
            "answer": item["answer"],
            "solution": item["solution"],
            "source": item["source"],
            "domain": item["domain"],
            "llama8b_solve_rate": item["llama8b_solve_rate"],
            "difficulty_bin": item["difficulty_bin"],
            "bigmath_index": item.get("bigmath_index"),
            "numina_index": item.get("numina_index")
        }
        formatted_problems.append(formatted_item)
    
    # Save the sampled problems
    with open(output_file, "w") as f:
        json.dump(formatted_problems, f, indent=2)
    
    print(f"\nSampled {len(formatted_problems)} problems across {len(bins)} difficulty bins")
    print(f"Saved to {output_file}")
    
    return formatted_problems

if __name__ == "__main__":
    # You can customize these parameters as needed
    sample_joined_problems(
        input_file="data/joined_math_datasets.json",
        output_file="data/joined_samples.json",
        samples_per_bin=10,
        seed=42
    ) 