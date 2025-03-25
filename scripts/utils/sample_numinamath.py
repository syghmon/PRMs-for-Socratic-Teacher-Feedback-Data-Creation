import json
import random
import os
import pathlib
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

# Get the project root directory (parent of scripts)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()

# Load environment variables from .env file
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

def sample_numinamath(output_file="data/numina_samples.json", sample_size=100, seed=None):
    # Authenticate with Hugging Face
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        print("Authenticating with Hugging Face...")
        login(token=hf_token)
    else:
        print("Warning: No Hugging Face token found in environment variables.")
    
    # Ensure output_file path is handled properly
    if not os.path.isabs(output_file):
        output_file = os.path.join(PROJECT_ROOT, output_file)
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Optionally set a random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Load the NuminaMath-1.5 dataset
    print("Loading dataset...")
    dataset = load_dataset("AI-MO/NuminaMath-1.5", split="train")
    print(f"Dataset loaded successfully with {len(dataset)} items")
    
    # Filter the dataset based on the requirements
    filtered_indices = []
    for idx, item in enumerate(dataset):
        # Check if problem is valid and solution is valid
        if (item.get("problem_is_valid", False) and 
            item.get("solution_is_valid", False) and 
            item.get("answer", "") and  # Answer must not be empty
            not is_proof(item.get("answer", ""))):  # Answer should not be a proof
            filtered_indices.append(idx)
    
    print(f"Found {len(filtered_indices)} items after filtering")
    
    if not filtered_indices:
        print("No items match the filtering criteria. Exiting.")
        return
    
    # Sample items randomly
    count = min(sample_size, len(filtered_indices))
    chosen_indices = random.sample(filtered_indices, count)
    
    sampled_items = []
    for idx in chosen_indices:
        item = dataset[idx]
        
        # Create a new item with the desired format
        formatted_item = {
            "problem": item.get("problem", ""),
            "answer": item.get("answer", ""),
            "solution": item.get("solution", ""),
            "source": item.get("source", "numinamath"),
            "domain": [item.get("problem_type", "Mathematics")] if item.get("problem_type") else ["Mathematics"],
            "llama8b_solve_rate": None,
            "difficulty_bin": None,
            "dataset_index": idx  # store original index for reference
        }
        sampled_items.append(formatted_item)
    
    # Save the sampled items to a JSON file
    with open(output_file, "w") as f:
        json.dump(sampled_items, f, indent=2)
    print(f"Saved {len(sampled_items)} sampled problems to {output_file}")

def is_proof(text):
    """
    Check if the answer looks like a proof rather than a direct answer.
    This is a simple heuristic and may need refinement.
    """
    # Convert to lowercase for easier comparison
    lower_text = text.lower()
    
    # Check if it contains proof-like phrases
    proof_indicators = ["proof", "suppose", "let's prove", "we need to show", 
                        "we will prove", "let us prove", "we can prove",
                        "to prove", "therefore", "thus we have proven"]
    
    # Check if the text is too long (proofs tend to be longer)
    if len(text.split()) > 100:
        return True
    
    # Check for proof indicators
    for indicator in proof_indicators:
        if indicator in lower_text:
            return True
    
    return False

# Run the sampling when script is executed
if __name__ == "__main__":
    sample_numinamath() 