#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from collections import defaultdict

# Hugging Face dataset name
HF_DATASET_NAME = "Syghmon/math-inference-results"

def load_data():
    """Load the dataset from Hugging Face"""
    print(f"Loading data from Hugging Face dataset: {HF_DATASET_NAME}")
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(HF_DATASET_NAME)
        
        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame(dataset['train'])
        print(f"Loaded {len(df)} samples")
        return df
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        print("Trying to fall back to local file...")
        
        # Fallback to local file if Hugging Face fails
        try:
            with open("data/predictions.json", 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            print(f"Loaded {len(df)} samples from local file")
            return df
        except Exception as local_e:
            print(f"Error loading local file: {local_e}")
            return None

def analyze_by_difficulty(df):
    """Analyze model performance by difficulty bin compared to llama8b"""
    # Check if necessary columns exist
    required_cols = ['llama8b_solve_rate', 'difficulty_bin']
    correctness_col = None
    
    # Look for correctness column
    potential_correctness_cols = [
        'responses_correctness', 
        'responses_extracted_answers_correctness'
    ]
    
    # Try all potential correctness columns
    for col in potential_correctness_cols:
        if col in df.columns:
            correctness_col = col
            print(f"Found correctness column: {correctness_col}")
            break
    
    # Check for other required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols or not correctness_col:
        if not correctness_col:
            print("Error: No correctness column found in the dataset")
            print("Available columns:", df.columns.tolist())
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
        return None, None

    # Group by difficulty bin
    results = defaultdict(dict)
    difficulty_bins = sorted(df['difficulty_bin'].unique())
    
    print("\nAnalyzing performance by difficulty bin:")
    print("=" * 80)
    print(f"{'Difficulty Bin':<15} {'Your Model':<15} {'llama8b':<15} {'Difference':<15}")
    print("-" * 80)
    
    for bin_name in difficulty_bins:
        bin_df = df[df['difficulty_bin'] == bin_name]
        bin_count = len(bin_df)
        
        # Calculate success rates
        if isinstance(bin_df[correctness_col].iloc[0], list):
            # If correctness is a list (multiple samples per problem)
            your_model_rate = sum(any(correct for correct in sample if correct is not None) 
                                for sample in bin_df[correctness_col]) / bin_count
        else:
            # If correctness is a single boolean per problem
            your_model_rate = bin_df[correctness_col].mean()
            
        llama_rate = bin_df['llama8b_solve_rate'].mean()
        difference = your_model_rate - llama_rate
        
        results[bin_name] = {
            'your_model': your_model_rate,
            'llama8b': llama_rate,
            'difference': difference,
            'count': bin_count
        }
        
        print(f"{bin_name:<15} {your_model_rate:.2%}<{bin_count:>3}> {llama_rate:.2%}<{bin_count:>3}> {difference:+.2%}")
    
    return results, difficulty_bins

def plot_results(results, difficulty_bins):
    """Create visualizations of the results"""
    # Extract data for plotting
    your_model_rates = [results[bin_name]['your_model'] for bin_name in difficulty_bins]
    llama_rates = [results[bin_name]['llama8b'] for bin_name in difficulty_bins]
    differences = [results[bin_name]['difference'] for bin_name in difficulty_bins]
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Bar plot comparing performance
    x = np.arange(len(difficulty_bins))
    width = 0.35
    
    plt.subplot(2, 1, 1)
    plt.bar(x - width/2, your_model_rates, width, label='Your Model')
    plt.bar(x + width/2, llama_rates, width, label='llama8b')
    
    plt.xlabel('Difficulty Bin')
    plt.ylabel('Solve Rate')
    plt.title('Model Performance by Difficulty Bin')
    plt.xticks(x, difficulty_bins, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot performance difference
    plt.subplot(2, 1, 2)
    colors = ['green' if diff >= 0 else 'red' for diff in differences]
    plt.bar(x, differences, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.xlabel('Difficulty Bin')
    plt.ylabel('Difference (Your Model - llama8b)')
    plt.title('Performance Difference')
    plt.xticks(x, difficulty_bins, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("\nSaved visualization to 'model_comparison.png'")
    
    # Show overall statistics
    overall_your_model = np.mean(your_model_rates)
    overall_llama = np.mean(llama_rates)
    overall_diff = overall_your_model - overall_llama
    
    print("\nOverall Performance:")
    print(f"Your Model: {overall_your_model:.2%}")
    print(f"llama8b:   {overall_llama:.2%}")
    print(f"Difference: {overall_diff:+.2%}")

def main():
    # Load data
    df = load_data()
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Print columns to help debug
    print("\nAvailable columns:", df.columns.tolist())
    
    # Analyze performance by difficulty
    results, difficulty_bins = analyze_by_difficulty(df)
    
    # Plot results
    if results:
        plot_results(results, difficulty_bins)
    else:
        print("Analysis failed. Cannot generate plots.")

if __name__ == "__main__":
    main() 