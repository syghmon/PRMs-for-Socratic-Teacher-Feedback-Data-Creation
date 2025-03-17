#!/usr/bin/env python3
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import os

def test_model(model_name, examples=None, custom_system_prompt=None):
    """Test a model with specific examples and print outputs."""
    print(f"\nTesting model: {model_name}")
    
    # Initialize model and tokenizer
    print(f"Loading model and tokenizer...")
    llm = LLM(model=model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    
    # Default system prompt if none provided
    system_prompt = custom_system_prompt or "You are a helpful AI assistant."
    
    # Default test examples if none provided
    if not examples:
        examples = [
            "What is 84 * 3 / 2?",
            "Tell me an interesting fact about the universe!"
        ]
    
    # Set up deterministic sampling
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic output for reproducibility
        top_p=1.0,
        max_tokens=256
    )
    
    results = []
    
    for i, example in enumerate(examples, 1):
        print(f"\n----- Example {i} -----")
        print(f"Input: {example}")
        
        # Format with chat template
        token_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example}
            ],
            add_bos=True,
            add_generation_prompt=True,
            tokenize=True
        )
        
        # Generate response
        outputs = llm.generate(
            prompt_token_ids=[token_ids],
            sampling_params=sampling_params
        )
        
        response = outputs[0].outputs[0].text
        print(f"Output: {response}")
        
        results.append({
            "input": example,
            "output": response
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test LLM models with examples from HuggingFace")
    
    parser.add_argument("--model", type=str, default="huggingfaceTB/SmolLM2-1.7B-Instruct",
                        help="Model name or path to test")
    
    parser.add_argument("--system-prompt", type=str, 
                       help="Custom system prompt to use")
    
    parser.add_argument("--examples", type=str,
                       help="JSON file with examples to test (array of strings)")
    
    parser.add_argument("--save", type=str,
                       help="Save results to this JSON file")
    
    args = parser.parse_args()
    
    # Load examples if provided
    examples = None
    if args.examples and os.path.exists(args.examples):
        with open(args.examples, "r") as f:
            examples = json.load(f)
    
    # Run the tests
    results = test_model(
        args.model, 
        examples=examples,
        custom_system_prompt=args.system_prompt
    )
    
    # Save results if requested
    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save}")

if __name__ == "__main__":
    main() 