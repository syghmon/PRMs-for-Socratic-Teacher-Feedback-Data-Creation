import argparse
import os
import subprocess
import sys
import json
from pathlib import Path
from datasets import Dataset, load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

def main():
    # Authenticate with Hugging Face if pushing to hub
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        print("Authenticating with Hugging Face...")
        login(token=hf_token)
    else:
        print("Warning: No Hugging Face token found in environment variables.")
        
    parser = argparse.ArgumentParser(description="Run inference and evaluate math problems.")
    # Inference parameters
    parser.add_argument("--input", "-i", type=str, default="data/samples.json",
                        help="Path to the JSON file of sampled problems (output of sampling script).")
    parser.add_argument("--output", "-o", type=str, default="data/predictions.json",
                        help="Path to save the JSON file with model predictions.")
    parser.add_argument("--model", "-m", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Hugging Face model name or local path for the model.")
    parser.add_argument("--local_model", action="store_true",
                        help="If set, treats the model argument as a local path instead of a HuggingFace model ID.")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum number of tokens to generate for each answer.")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature for generation (0.6 is recommended).")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of samples to generate for each problem.")
    
    # HuggingFace dataset parameters
    parser.add_argument("--hf_dataset_name", type=str, required=True,
                        help="Name for the HuggingFace dataset (e.g., 'username/dataset-name').")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether to push the predictions to the HuggingFace Hub.")
    
    # Evaluation parameters
    parser.add_argument("--evaluate", action="store_true",
                        help="Whether to run evaluation after inference.")
    parser.add_argument("--response_column_name", type=str, default="responses",
                        help="The name of the column containing model responses.")
    parser.add_argument("--ground_truth_answer_column_name", type=str, default="final_answer",
                        help="The name of the column containing ground truth answers.")
    parser.add_argument("--use_llm_judge_backup", action="store_true",
                        help="Whether to use LLM judge as backup in evaluation.")
    parser.add_argument("--semaphore_limit", type=int, default=20,
                        help="The maximum number of concurrent requests to the evaluator.")
    
    # Add skip_inference parameter
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip the inference step and use existing predictions file.")
    
    args = parser.parse_args()
    
    # Step 1: Run inference to generate samples
    if not args.skip_inference:
        print("=== Step 1: Running inference to generate samples ===")
        inference_cmd = [
            "python", "scripts/inference.py",
            "--input", args.input,
            "--output", args.output,
            "--model", args.model,
            "--max_tokens", str(args.max_tokens),
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--num_samples", str(args.num_samples),
            "--format_for_hf"
        ]
        
        if args.local_model:
            inference_cmd.append("--local_model")
        
        print(f"Running command: {' '.join(inference_cmd)}")
        try:
            subprocess.run(inference_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running inference: {e}")
            sys.exit(1)
    else:
        print("=== Step 1: Skipping inference step as requested ===")
        print(f"Using existing predictions file: {args.output}")
    
    # Step 2: Upload to HuggingFace Hub
    if args.push_to_hub:
        print("=== Step 2: Uploading to HuggingFace Hub ===")
        try:
            # Load the predictions
            with open(args.output, 'r') as f:
                predictions = json.load(f)
            
            # Convert to HuggingFace dataset
            dataset = Dataset.from_list(predictions)
            print(f"Dataset info: {dataset}")
            
            # Push to hub
            dataset.push_to_hub(args.hf_dataset_name, private=True)
            print(f"Successfully pushed dataset to {args.hf_dataset_name}")
        except Exception as e:
            print(f"Error uploading to HuggingFace Hub: {e}")
            sys.exit(1)
    
    # Step 3: Evaluate the predictions
    if args.evaluate:
        print("=== Step 3: Evaluating predictions ===")
        evaluate_cmd = [
            "python", "scripts/evaluate_responses.py",
            "--predictions_dataset_name", args.hf_dataset_name,
            "--response_column_name", args.response_column_name,
            "--ground_truth_answer_column_name", args.ground_truth_answer_column_name,
            "--semaphore_limit", str(args.semaphore_limit)
        ]
        
        if args.use_llm_judge_backup:
            evaluate_cmd.append("--use_llm_judge_backup")
        
        print(f"Running command: {' '.join(evaluate_cmd)}")
        try:
            subprocess.run(evaluate_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation: {e}")
            sys.exit(1)
    
    print("=== Complete ===")

if __name__ == "__main__":
    main() 