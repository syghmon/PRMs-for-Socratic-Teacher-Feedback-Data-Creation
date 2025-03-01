import json
import random
from datasets import load_dataset

def sample_problems(output_file="samples.json", seed=None):
    # Optionally set a random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Load the Big-Math-RL-Verified dataset (ensure you have access to it)
    dataset = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
    
    # Define the solve rate buckets (percent ranges)
    buckets = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
               (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    
    sampled_items = []
    for low, high in buckets:
        # Filter problems with solve_rate in [low, high)
        if high == 1.01:
            # Include 100% in the last bucket
            bucket_data = [idx for idx, item in enumerate(dataset) 
                           if item["llama8b_solve_rate"] is not None 
                           and low <= item["llama8b_solve_rate"] <= 1]
        else:
            bucket_data = [idx for idx, item in enumerate(dataset) 
                           if item["llama8b_solve_rate"] is not None 
                           and low <= item["llama8b_solve_rate"] < high]
        if not bucket_data:
            print(f"No samples found in range {low}-{high}%. Skipping this range.")
            continue
        
        # If fewer than 10 items exist in the bucket, take all; otherwise sample 10
        count = 10 if len(bucket_data) >= 10 else len(bucket_data)
        chosen_indices = random.sample(bucket_data, count)
        
        for idx in chosen_indices:
            item = dataset[idx]
            # Add difficulty bin info for clarity
            item["difficulty_bin"] = f"{low}-{high if high != 1.01 else 1}%"
            item["dataset_index"] = idx  # store original index for reference
            sampled_items.append(item)
    
    # If we got more or less than 100 (e.g., some bucket had <10 items), adjust or notify
    total = len(sampled_items)
    print(f"Total sampled problems: {total}")
    if total < 100:
        print("Warning: Less than 100 samples were collected due to insufficient data in some ranges.")
    elif total > 100:
        # If some bucket had extra items, trim the list to 100 (though this should not happen with default logic)
        sampled_items = sampled_items[:100]
    
    # Save the sampled items to a JSON file
    with open(output_file, "w") as f:
        json.dump(sampled_items, f, indent=2)
    print(f"Saved sampled problems to {output_file}")

# Run the sampling when script is executed
if __name__ == "__main__":
    sample_problems()
