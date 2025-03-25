import json
import os
import pathlib
import re
import string
import gc
import hashlib
from difflib import SequenceMatcher
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

# Get the project root directory (parent of scripts)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()

# Load environment variables from .env file
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

def normalize_problem(text):
    """
    Normalize the problem text for exact matching.
    Remove problem numbers at the beginning (like "4.") and extra markers.
    Normalize whitespace but preserve LaTeX.
    """
    # Remove numbers at the beginning of problems (e.g., "4.")
    text = re.sub(r'^\s*\d+\.\s*', '', text)
    
    # Remove trailing periods and extraneous markers
    text = re.sub(r'\.$', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_problem_content(text):
    """
    Extract the main mathematical content of the problem,
    removing any metadata, problem numbers, etc.
    """
    # Remove numbers at the beginning of problems (e.g., "4.")
    text = re.sub(r'^\s*\d+\.\s*', '', text)
    
    # Remove trailing periods and extraneous markers
    text = re.sub(r'\.$', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def generate_problem_hash(text):
    """
    Generate a hash of the normalized problem text for faster matching.
    """
    normalized = extract_problem_content(text)
    return hashlib.md5(normalized.encode()).hexdigest()

def join_datasets(output_file="data/joined_math_datasets.json", bigmath_sample_size=None, seed=None, debug_mode=False):
    """
    Join BigMath and NuminaMath datasets based on problem similarity.
    Keep BigMath metadata but use NuminaMath solutions.
    Only uses fast hash-based exact matching.
    
    Args:
        output_file: Path to save the joined dataset
        bigmath_sample_size: Limit the number of BigMath problems to process (for testing)
        seed: Random seed for reproducibility
        debug_mode: If True, outputs additional debug information
    """
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
    
    # Temporary file to store BigMath samples
    temp_bigmath_file = os.path.join(os.path.dirname(output_file), "temp_bigmath_samples.json")
    
    # PHASE 1: Load and process BigMath dataset (or use existing temp file)
    if os.path.exists(temp_bigmath_file) and not debug_mode:
        print(f"PHASE 1: Using existing BigMath samples from {temp_bigmath_file}")
        with open(temp_bigmath_file, "r") as f:
            bigmath_samples = json.load(f)
        print(f"Loaded {len(bigmath_samples)} BigMath samples")
    else:
        print("PHASE 1: Loading BigMath dataset...")
        bigmath_dataset = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
        print(f"BigMath dataset loaded with {len(bigmath_dataset)} items")
        
        # Sample from BigMath
        if bigmath_sample_size is not None:
            bigmath_indices = list(range(min(bigmath_sample_size, len(bigmath_dataset))))
        else:
            bigmath_indices = list(range(len(bigmath_dataset)))
        
        # Extract relevant data from BigMath and generate problem hashes
        bigmath_samples = []
        for idx in bigmath_indices:
            item = bigmath_dataset[idx]
            # Only include problems with solve rates for proper difficulty categorization
            if item.get("llama8b_solve_rate") is not None:
                problem_text = extract_problem_content(item["problem"])
                problem_hash = generate_problem_hash(item["problem"])
                
                sample = {
                    "problem": item["problem"],
                    "normalized_problem": problem_text,
                    "answer": item["answer"],
                    "source": item["source"],
                    "domain": item["domain"],
                    "llama8b_solve_rate": item["llama8b_solve_rate"],
                    "difficulty_bin": get_difficulty_bin(item["llama8b_solve_rate"]),
                    "bigmath_index": idx,
                    "problem_hash": problem_hash
                }
                bigmath_samples.append(sample)
        
        print(f"Extracted {len(bigmath_samples)} BigMath samples with solve rates")
        
        # Save BigMath samples to temporary file
        if not debug_mode:
            with open(temp_bigmath_file, "w") as f:
                json.dump(bigmath_samples, f)
        
        # Clear BigMath dataset from memory
        del bigmath_dataset
        gc.collect()
        print("Freed BigMath dataset from memory")
    
    # Create a dictionary of problem hashes for exact matching
    bigmath_hash_dict = {item["problem_hash"]: i for i, item in enumerate(bigmath_samples)}
    
    # PHASE 2: Load and process NuminaMath dataset
    print("\nPHASE 2: Loading NuminaMath dataset...")
    numina_dataset = load_dataset("AI-MO/NuminaMath-1.5", split="train")
    print(f"NuminaMath dataset loaded with {len(numina_dataset)} items")
    
    # Filter NuminaMath dataset based on requirements and match by hash
    print("Filtering NuminaMath dataset and finding hash matches...")
    exact_matches = []
    numina_matched_indices = set()
    bigmath_matched_indices = set()
    
    # Debug logging for the first few matches if debug_mode is enabled
    debug_samples = []
    
    # Process NuminaMath items one by one to prevent memory issues
    for idx, item in enumerate(numina_dataset):
        # Print progress every 50,000 items
        if idx % 50000 == 0:
            print(f"Processed {idx}/{len(numina_dataset)} NuminaMath items, found {len(exact_matches)} matches so far")
        
        # Check filtering criteria first
        if not (item.get("problem_is_valid", False) and 
                item.get("solution_is_valid", False) and 
                item.get("answer", "") and  # Answer must not be empty
                not is_proof(item.get("answer", "")) and  # Answer should not be a proof
                item.get("solution", "")):  # Solution must not be empty
            continue
        
        # Extract and normalize the problem content
        numina_problem_text = extract_problem_content(item["problem"])
        
        # Calculate hash for exact matching
        problem_hash = generate_problem_hash(item["problem"])
        
        # For debugging
        if debug_mode and len(debug_samples) < 10:
            debug_samples.append({
                "numina_index": idx,
                "numina_problem": item["problem"],
                "numina_normalized": numina_problem_text,
                "numina_hash": problem_hash,
                "numina_answer": item["answer"],
                "numina_solution": item["solution"]
            })
        
        # Check if this problem matches any BigMath problem
        if problem_hash in bigmath_hash_dict:
            bigmath_idx = bigmath_hash_dict[problem_hash]
            
            # Verify the bigmath index hasn't been matched already
            if bigmath_idx not in bigmath_matched_indices:
                bigmath_item = bigmath_samples[bigmath_idx]
                
                # Additional verification - compare normalized problems to ensure match quality
                similarity = SequenceMatcher(None, numina_problem_text, bigmath_item["normalized_problem"]).ratio()
                
                # Skip poor matches
                if similarity < 0.7:
                    if debug_mode:
                        print(f"Low similarity match skipped ({similarity}):")
                        print(f"BigMath: {bigmath_item['problem']}")
                        print(f"NuminaMath: {item['problem']}\n")
                    continue
                
                # Create joined item with both sources for verification
                joined_item = {
                    "problem": bigmath_item["problem"],
                    "answer": bigmath_item["answer"],
                    "solution": item["solution"],
                    "source": bigmath_item["source"],
                    "domain": bigmath_item["domain"],
                    "llama8b_solve_rate": bigmath_item["llama8b_solve_rate"],
                    "difficulty_bin": bigmath_item["difficulty_bin"],
                    "bigmath_index": bigmath_item["bigmath_index"],
                    "numina_index": idx,
                    "match_type": "exact_hash",
                    "similarity_score": similarity,
                    "numina_problem": item["problem"],  # For verification
                    "numina_answer": item["answer"]     # For verification
                }
                
                # For debugging, add the normalized versions and hashes
                if debug_mode:
                    joined_item["bigmath_normalized"] = bigmath_item["normalized_problem"]
                    joined_item["numina_normalized"] = numina_problem_text
                    joined_item["bigmath_hash"] = bigmath_item["problem_hash"]
                    joined_item["numina_hash"] = problem_hash
                
                exact_matches.append(joined_item)
                bigmath_matched_indices.add(bigmath_idx)
                numina_matched_indices.add(idx)
                
                # Debug logging for the first few matches
                if debug_mode and len(exact_matches) <= 5:
                    print(f"\nMatch {len(exact_matches)}:")
                    print(f"BigMath: {bigmath_item['problem']}")
                    print(f"NuminaMath: {item['problem']}")
                    print(f"Similarity: {similarity}")
                    print(f"BigMath answer: {bigmath_item['answer']}")
                    print(f"NuminaMath answer: {item['answer']}")
                    print(f"NuminaMath solution: {item['solution'][:100]}...\n")
    
    print(f"Found {len(exact_matches)} exact hash matches")
    
    # Save joined items to output file
    with open(output_file, "w") as f:
        json.dump(exact_matches, f, indent=2)
    print(f"Saved joined dataset to {output_file}")
    
    # Save debug samples if in debug mode
    if debug_mode and debug_samples:
        debug_file = os.path.join(os.path.dirname(output_file), "debug_samples.json")
        with open(debug_file, "w") as f:
            json.dump(debug_samples, f, indent=2)
        print(f"Saved debug samples to {debug_file}")
    
    # Clear data from memory
    del numina_dataset
    gc.collect()
    print("Freed NuminaMath dataset from memory")
    
    return len(exact_matches)

def get_difficulty_bin(solve_rate):
    """
    Determine the difficulty bin based on solve rate.
    """
    if solve_rate is None:
        return None
    
    buckets = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
               (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    
    for low, high in buckets:
        if high == 1.01:
            if low <= solve_rate <= 1:
                return f"{low}-{1}%"
        else:
            if low <= solve_rate < high:
                return f"{low}-{high}%"
    
    return None

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

def test_problem_matching():
    """
    Test function to validate the problem matching logic with a few examples.
    """
    print("Testing problem matching...")
    
    # Example problems
    test_problems = [
        # Same problems, different formatting or problem numbers
        {
            "problem1": "4. Find x such that 2x + 3 = 7.",
            "problem2": "Find x such that 2x + 3 = 7",
            "should_match": True
        },
        {
            "problem1": "$\\frac{|x+1|}{|x^{2}-3 x-4|}=\\frac{2}{|x-1|}$.",
            "problem2": "4. $\\frac{|x+1|}{\\left|x^{2}-3 x-4\\right|}=\\frac{2}{|x-1|}$.",
            "should_match": True
        },
        # Different problems
        {
            "problem1": "Solve for x: 2x + 3 = 7.",
            "problem2": "Solve for x: 3x - 2 = 7.",
            "should_match": False
        },
        {
            "problem1": "$\\log _{b} N=\\frac{\\log _{a} N}{\\log _{a} b}$",
            "problem2": "$\\frac{|x+1|}{|x^{2}-3 x-4|}=\\frac{2}{|x-1|}$",
            "should_match": False
        }
    ]
    
    for i, test in enumerate(test_problems):
        problem1 = test["problem1"]
        problem2 = test["problem2"]
        expected = test["should_match"]
        
        # Generate hashes
        hash1 = generate_problem_hash(problem1)
        hash2 = generate_problem_hash(problem2)
        
        # Check if hashes match as expected
        actual = (hash1 == hash2)
        
        # Calculate similarity for reference
        similarity = SequenceMatcher(None, 
                                   extract_problem_content(problem1), 
                                   extract_problem_content(problem2)).ratio()
        
        print(f"\nTest {i+1}:")
        print(f"Problem 1: {problem1}")
        print(f"Normalized: {extract_problem_content(problem1)}")
        print(f"Problem 2: {problem2}")
        print(f"Normalized: {extract_problem_content(problem2)}")
        print(f"Similarity: {similarity:.2f}")
        print(f"Hash 1: {hash1}")
        print(f"Hash 2: {hash2}")
        print(f"Expected match: {expected}")
        print(f"Actual match: {actual}")
        print(f"Test {'PASSED' if expected == actual else 'FAILED'}")
    
    print("\nFinished testing problem matching")

# Run the joining when script is executed
if __name__ == "__main__":
    # Test the problem matching logic
    test_problem_matching()
    
    # Run with debug mode to see detailed information about the first few matches
    print("\n" + "="*80 + "\n")
    join_datasets(debug_mode=True) 