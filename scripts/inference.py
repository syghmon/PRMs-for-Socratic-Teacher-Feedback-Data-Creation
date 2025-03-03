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
    parser = argparse.ArgumentParser(description="Run Qwen Distill (R1) inference on sampled math problems.")
    parser.add_argument("--input", "-i", type=str, default="data/samples.json",
                        help="Path to the JSON file of sampled problems (output of sampling script).")
    parser.add_argument("--output", "-o", type=str, default="data/predictions.json",
                        help="Path to save the JSON file with model predictions.")
    parser.add_argument("--model", "-m", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Hugging Face model name or local path for the model.")
    parser.add_argument("--local_model", action="store_true",
                        help="If set, treats the model argument as a local path instead of a HuggingFace model ID.")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate for each answer.")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature for generation (0.6 is recommended).")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling parameter.")
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
    for item in samples:
        problem_text = item["problem"]
        # Construct the prompt with instructions for reasoning and answer format
        prompt = (f"{problem_text}\n\n"
                  "Please reason step by step, and put your final answer in \\boxed{}.")
        prompts.append(prompt)
    
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
        
        # Run inference in batches if needed
        print(f"Generating answers for {len(prompts)} problems...")
        # You can split prompts into chunks if memory is a concern. Here we do it in one batch.
        outputs = llm.generate(prompts, sampling_params)
        
        # Collect results
        results = []
        for item, output in zip(samples, outputs):
            # Each output corresponds to one prompt
            generated_text = output.outputs[0].text  # the model's full response
            result_entry = {
                "problem": item.get("problem"),
                "ground_truth": item.get("answer"),
                "model_answer": generated_text
            }
            # Include any metadata for reference (optional)
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
