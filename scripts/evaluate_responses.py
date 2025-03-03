import argparse
import asyncio
from datasets import load_dataset
import random
import json
import os
from functools import partial
from tqdm.asyncio import tqdm_asyncio
from typing import Union

from math_eval import MathEvaluator, is_correct_no_judge, get_answer_expr, last_boxed_only_string, remove_boxed

async def rate_limit_is_correct(evaluator: MathEvaluator, answer: str, pred: str, sem: asyncio.Semaphore):
    """
    Asynchronously checks if the predicted answer is correct, with rate limiting.
    """
    async with sem:
        return await evaluator.is_correct(answer, pred)

async def evaluate_preds_async(preds: Union[list[list],list], ground_truth_answers: list, evaluator: MathEvaluator, sem: asyncio.Semaphore = None):
    """
    Asynchronously evaluates predictions against ground truth answers using a provided evaluator.

    Args:
        preds (Union[list[list], list]): A list of predictions or a list of lists of predictions.
        ground_truth_answers (list): A list of ground truth answers corresponding to the predictions.
        evaluator (MathEvaluator): An instance of MathEvaluator used to evaluate the predictions.
        sem (asyncio.Semaphore, optional): An optional semaphore to limit the number of concurrent evaluations. Defaults to None.

    Returns:
        list: A list of evaluation results.

    """
    print("Evaluating predictions...")
    if isinstance(preds[0], list):
        tasks = []
        for pred, answer in zip(preds, ground_truth_answers):
            tasks.extend([asyncio.create_task(rate_limit_is_correct(evaluator, answer, p, sem)) for p in pred])
    else:
        tasks = [asyncio.create_task(rate_limit_is_correct(evaluator, answer, pred, sem)) for pred, answer in zip(preds, ground_truth_answers)]
    result = await tqdm_asyncio.gather(*tasks)
    return result

def extract_answers(row, evaluator: MathEvaluator, response_column_name: str, predicted_answer_column_name: str):
    """Extracts answers from the response column and adds them as a separate column."""
    if isinstance(row[response_column_name], list):
        row[predicted_answer_column_name] = [evaluator.get_answer_expr(p) for p in row[response_column_name]]
    else:
        row[predicted_answer_column_name] = evaluator.get_answer_expr(row[response_column_name])
    return row

def extract_and_evaluate_answers(
        row,
        response_column_name: str,
        predicted_answer_column_name: str,
        predicted_answer_correctness_column_name: str,
        ground_truth_answer_column_name: str,
        ):
    """Extracts answers from the response column and evaluates correctness."""
    if isinstance(row[response_column_name], list):
        row[predicted_answer_column_name] = [get_answer_expr(p) for p in row[response_column_name]]
        row[predicted_answer_correctness_column_name] = [is_correct_no_judge(row[ground_truth_answer_column_name], p) for p in row[predicted_answer_column_name]]
    else:
        row[predicted_answer_column_name] = get_answer_expr(row[response_column_name])
        row[predicted_answer_correctness_column_name] = is_correct_no_judge(row[ground_truth_answer_column_name], row[predicted_answer_column_name])
    return row

def inspect_sample_answers(dataset, response_column_name, predicted_answer_column_name, ground_truth_answer_column_name, output_file, num_samples=5):
    """
    Saves a sample of original responses, extracted answers, and ground truth answers to a JSON file for manual inspection.
    
    Args:
        dataset: The dataset containing the responses and answers.
        response_column_name: The name of the column containing the original responses.
        predicted_answer_column_name: The name of the column containing the extracted/predicted answers.
        ground_truth_answer_column_name: The name of the column containing the ground truth answers.
        output_file: Path to the output JSON file.
        num_samples: Number of samples to save. Default is 5.
    """
    
    dataset_size = len(dataset)
    
    # Group samples by problem to ensure diversity
    problem_indices = {}
    for idx in range(dataset_size):
        if 'problem' in dataset[idx]:
            problem_key = dataset[idx]['problem'][:100]  # Use first 100 chars as key
            if problem_key not in problem_indices:
                problem_indices[problem_key] = []
            problem_indices[problem_key].append(idx)
    
    # If we have problem groupings, sample from different problems
    if problem_indices and len(problem_indices) >= num_samples:
        # Pick one sample from each problem, up to num_samples
        problems = list(problem_indices.keys())
        selected_problems = random.sample(problems, num_samples)
        indices = [random.choice(problem_indices[problem]) for problem in selected_problems]
    else:
        # Fall back to random sampling if we can't ensure problem diversity
        if dataset_size <= num_samples:
            indices = list(range(dataset_size))
        else:
            indices = random.sample(range(dataset_size), num_samples)
    
    samples = []
    for idx in indices:
        # Extract the boxed part from the original response
        original_response = dataset[idx][response_column_name]
        extracted_answer = dataset[idx][predicted_answer_column_name]
        ground_truth = dataset[idx][ground_truth_answer_column_name]
        
        # For multiple responses (e.g., in a list)
        if isinstance(original_response, list):
            boxed_answers = []
            for resp in original_response:
                try:
                    boxed = last_boxed_only_string(resp)
                    if boxed:
                        boxed_answers.append(boxed)
                    else:
                        boxed_answers.append("No boxed answer found")
                except Exception:
                    boxed_answers.append("Error extracting boxed answer")
            
            # Check correctness of each answer
            correctness = []
            if isinstance(extracted_answer, list):
                for ans in extracted_answer:
                    correctness.append(is_correct_no_judge(ground_truth, ans))
            else:
                correctness = [False] * len(boxed_answers)
                
            sample = {
                "sample_id": idx,
                "responses": [
                    {
                        "boxed_answer": box,
                        "extracted_answer": ext if isinstance(extracted_answer, list) and i < len(extracted_answer) else "Error",
                        "is_correct": cor if i < len(correctness) else False
                    }
                    for i, (box, ext, cor) in enumerate(zip(
                        boxed_answers, 
                        extracted_answer if isinstance(extracted_answer, list) else [None] * len(boxed_answers), 
                        correctness
                    ))
                ],
                "ground_truth_answer": ground_truth,
                "problem": dataset[idx].get('problem', 'Problem not available')
            }
        else:
            # Single response
            try:
                boxed = last_boxed_only_string(original_response)
                if not boxed:
                    boxed = "No boxed answer found"
            except Exception:
                boxed = "Error extracting boxed answer"
            
            # Check correctness
            is_correct = is_correct_no_judge(ground_truth, extracted_answer)
            
            sample = {
                "sample_id": idx,
                "boxed_answer": boxed,
                "extracted_answer": extracted_answer,
                "ground_truth_answer": ground_truth,
                "is_correct": is_correct,
                "problem": dataset[idx].get('problem', 'Problem not available')
            }
        
        samples.append(sample)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump({"samples": samples}, f, indent=2)
    
    print(f"\nSaved {len(samples)} sample answers to {output_file}")
    print("Format: Each sample shows the boxed answer (final answer), the extracted answer after processing, and the ground truth for comparison.")

async def async_main(args):
    # define the column names

    response_column_name = args.response_column_name
    ground_truth_answer_column_name = args.ground_truth_answer_column_name

    # define the answer and correctness columns
    predicted_answer_column_name = f"{args.response_column_name}_extracted_answers"
    predicted_answer_correctness_column_name = f"{args.response_column_name}_correctness"

    # load the dataset
    preds_dataset = load_dataset(args.predictions_dataset_name, split=args.dataset_split)

    # Make sure we can fulfill the evaluation requirements
    if args.calculate_greedy_accuracy:
        assert "greedy_correct" in preds_dataset.column_names, "Predictions dataset does not contain greedy_correct column"

    evaluator = MathEvaluator()

    # Extract the answers from responses
    if predicted_answer_column_name in preds_dataset.column_names or predicted_answer_correctness_column_name in preds_dataset.column_names:
        # if answers are already extracted, or correctness is already calculated, skip extraction
        print("Answers already extracted. Skipping extraction")
    else:
        # extract the answers from the response column and add them as a separate column
        print(f"Extracting answers to {predicted_answer_column_name}...")
        partial_extract_answers = partial(
            extract_answers,
            evaluator=evaluator,
            response_column_name=response_column_name,
            predicted_answer_column_name=predicted_answer_column_name
        )
        # TODO: Rewrite the math_evaluator.is_correct() function so that it can live outside of the MathEvaluator object
        # otherwise, this can only run with num_proc=1
        # HuggingFace datasets cannot pickle the MathEvaluator object
        # returns error: "TypeError: cannot pickle 'SSLContext' object"
        preds_dataset = preds_dataset.map(partial_extract_answers)

    if predicted_answer_correctness_column_name in preds_dataset.column_names:
        # if correctness is already calculated, skip evaluation
        print("Correctness already calculated. Skipping evaluation")
    else:
        # calculate the correctness of predicted answers
        print(f"Evaluating correctness of {predicted_answer_column_name}...")
        sem = asyncio.Semaphore(args.semaphore_limit)
        answer_correctness = await evaluate_preds_async(
            preds=preds_dataset[predicted_answer_column_name],
            ground_truth_answers=preds_dataset[ground_truth_answer_column_name],
            evaluator=evaluator,
            sem=sem
        )
        
        # reshape the correctness list to match the shape of the predictions
        if isinstance(preds_dataset[predicted_answer_column_name][0], list):
            reshaped_answer_correctness = []
            cur_start = 0
            for i in range(len(preds_dataset[predicted_answer_column_name])):
                cur_len = len(preds_dataset[predicted_answer_column_name][i])
                reshaped_answer_correctness.append(answer_correctness[cur_start:cur_start+cur_len])
                cur_start += cur_len
            answer_correctness = reshaped_answer_correctness
        
        # add the correctness column to the dataset
        preds_dataset = preds_dataset.add_column(predicted_answer_correctness_column_name, answer_correctness)

    # Inspect a sample of answers if requested
    if args.inspect_samples:
        
        inspect_sample_answers(
            preds_dataset, 
            response_column_name, 
            predicted_answer_column_name, 
            ground_truth_answer_column_name,
            args.output_file,
            args.num_samples
        )

    # push the dataset back to the hub
    preds_dataset.push_to_hub(args.predictions_dataset_name, private=True)

def main(args):
    # define the column names

    response_column_name = args.response_column_name
    ground_truth_answer_column_name = args.ground_truth_answer_column_name

    # define the answer and correctness columns
    predicted_answer_column_name = f"{args.response_column_name}_extracted_answers"
    predicted_answer_correctness_column_name = f"{args.response_column_name}_correctness"

    # load the dataset
    preds_dataset = load_dataset(args.predictions_dataset_name, split=args.dataset_split)

    # Make sure we can fulfill the evaluation requirements
    if args.calculate_greedy_accuracy:
        assert "greedy_correct" in preds_dataset.column_names, "Predictions dataset does not contain greedy_correct column"

    # Extract answers and evaluate correctness in a single step
    print(f"Extracting and evaluating correctness of {predicted_answer_column_name}...")
    partial_extract_and_evaluate_answers = partial(
        extract_and_evaluate_answers,
        response_column_name=response_column_name,
        predicted_answer_column_name=predicted_answer_column_name,
        predicted_answer_correctness_column_name=predicted_answer_correctness_column_name,
        ground_truth_answer_column_name=ground_truth_answer_column_name
    )
    
    preds_dataset = preds_dataset.map(partial_extract_and_evaluate_answers, num_proc=64)

    # Inspect a sample of answers if requested
    if args.inspect_samples:
        
        inspect_sample_answers(
            preds_dataset, 
            response_column_name, 
            predicted_answer_column_name, 
            ground_truth_answer_column_name,
            args.output_file,
            args.num_samples
        )

    # push the dataset back to the hub
    preds_dataset.push_to_hub(args.predictions_dataset_name, private=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate rollouts")
    # Model and dataset arguments
    parser.add_argument("--predictions_dataset_name", type=str, required=True,
                        help="The name of the dataset containing the predictions.")
    parser.add_argument("--response_column_name", type=str, required=True,
                        help="The name of the column containing the model's responses.")
    parser.add_argument("--ground_truth_answer_column_name", type=str, required=True,
                        help="The name of the column containing the ground truth answers.")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--calculate_greedy_accuracy", action="store_true")

    # Evaluation config
    parser.add_argument("--use_llm_judge_backup", action="store_true",
                        help="Whether to use the LLM judge as a backup for evaluation.")
                        
    # Inspection arguments
    parser.add_argument("--inspect_samples", action="store_true",
                        help="Whether to inspect a sample of answers for manual review.")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to save for inspection.")
    parser.add_argument("--output_file", type=str, default="samples/answer_samples.json",
                        help="Path to the output JSON file for saving samples.")

    # system configuration
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes to use for data processing.")
    parser.add_argument("--semaphore_limit", type=int, default=20,
                        help="The maximum number of concurrent requests to the evaluator, when using LLM judge as backup.")

    args = parser.parse_args()

    if args.use_llm_judge_backup:
        asyncio.run(async_main(args))
    else:
        main(args)

# Example usage:
# No LLM-judge
# python3 evaluate_responses.py --predictions_dataset_name RLAIF/Big-Math-needs-llama3-8b-rollouts --response_column_name responses --ground_truth_answer_column_name final_answer
# With LLM-judge
# python3 evaluate_responses.py --predictions_dataset_name RLAIF/Big-Math-needs-llama3-8b-rollouts --response_column_name responses --ground_truth_answer_column_name final_answer --use_llm_judge_backup 