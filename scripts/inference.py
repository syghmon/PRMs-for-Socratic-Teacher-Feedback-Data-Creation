import json
import argparse
import os
import pathlib
import sys
from requests.exceptions import ConnectionError
from vllm import LLM, SamplingParams

# Get the project root directory (parent of scripts)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()

def main():
    parser = argparse.ArgumentParser(description="Run model inference on sampled math problems.")
    parser.add_argument("--input", "-i", type=str, default="data/samples.json",
                        help="Path to the JSON file of sampled problems (output of sampling script).")
    parser.add_argument("--output", "-o", type=str, default="data/predictions.json",
                        help="Path to save the JSON file with model predictions.")
    parser.add_argument("--model", "-m", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Hugging Face model name or local path for the model.")
    parser.add_argument("--local_model", action="store_true",
                        help="If set, treats the model argument as a local path instead of a HuggingFace model ID.")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate for each answer.")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature for generation (0.6 is recommended).")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate for each problem.")
    parser.add_argument("--format_for_hf", action="store_true",
                        help="Format output for HuggingFace datasets upload.")
    args = parser.parse_args()
    
    # Ensure input and output file paths are handled properly
    input_file = args.input
    if not os.path.isabs(input_file):
        input_file = os.path.join(PROJECT_ROOT, input_file)
    
    output_file = args.output
    if not os.path.isabs(output_file):
        output_file = os.path.join(PROJECT_ROOT, output_file)
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load the sampled problems
    with open(input_file, "r") as f:
        samples = json.load(f)
    
    # Prepare prompts for the model
    prompts = []
    prompt_id_map = {}  # To track which problem each prompt belongs to
    
    for idx, item in enumerate(samples):
        problem_text = item["problem"]
        # Construct the prompt with instructions for reasoning and answer format
        prompt = (f"{problem_text}\n\n"
                "<think>\n"
                "Please reason step by step, and put your final answer in \\boxed{}.")
        
        # For each problem, add it to prompts num_samples times
        for _ in range(args.num_samples):
            prompts.append(prompt)
            prompt_id_map[len(prompts) - 1] = idx  # Map this prompt to the original problem index
    
    # Initialize the model with vLLM
    print(f"Loading model '{args.model}'... (This may take a while for large models)")
    try:
        # If using a local model, verify the path exists
        if args.local_model and not os.path.exists(args.model):
            print(f"Error: Local model path '{args.model}' does not exist.")
            sys.exit(1)
            
        llm = LLM(model=args.model, trust_remote_code=True)
        
        # Set up sampling/generation parameters
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        
        # Run inference
        print(f"Generating {args.num_samples} samples for each of {len(samples)} problems...")
        outputs = llm.generate(prompts, sampling_params)
        
        if args.format_for_hf:
            # Format for HuggingFace datasets
            results = []
            
            for i, item in enumerate(samples):
                # Collect all responses for this problem
                problem_responses = []
                for j in range(args.num_samples):
                    output_idx = i * args.num_samples + j
                    if output_idx < len(outputs):
                        problem_responses.append(outputs[output_idx].outputs[0].text)
                
                result_entry = {
                    "problem": item.get("problem"),
                    "final_answer": item.get("answer"),  # Renamed to match evaluation script expectation
                    "responses": problem_responses  # List of all responses for this problem
                }
                
                # Include any metadata
                if "difficulty_bin" in item:
                    result_entry["difficulty_bin"] = item["difficulty_bin"]
                if "llama8b_solve_rate" in item:
                    result_entry["llama8b_solve_rate"] = item["llama8b_solve_rate"]
                
                results.append(result_entry)
        else:
            # Original format with multiple samples
            results = []
            for prompt_idx, output in enumerate(outputs):
                sample_idx = prompt_id_map[prompt_idx]
                item = samples[sample_idx]
                
                # Each output corresponds to one prompt
                generated_text = output.outputs[0].text  # the model's full response
                result_entry = {
                    "problem": item.get("problem"),
                    "ground_truth": item.get("answer"),
                    "model_answer": generated_text,
                    "sample_id": prompt_idx % args.num_samples  # Which sample this is for the problem
                }
                
                # Include any metadata
                if "difficulty_bin" in item:
                    result_entry["difficulty_bin"] = item["difficulty_bin"]
                if "llama8b_solve_rate" in item:
                    result_entry["llama8b_solve_rate"] = item["llama8b_solve_rate"]
                
                results.append(result_entry)
        
        # Save results to output file
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved model outputs to {output_file}")
        
    except ConnectionError as e:
        print(f"Connection error when trying to download model: {e}")
        print("This might be due to network issues or proxy configuration.")
        print("Try using a local model with the --local_model flag or check your network settings.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during model loading or inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
