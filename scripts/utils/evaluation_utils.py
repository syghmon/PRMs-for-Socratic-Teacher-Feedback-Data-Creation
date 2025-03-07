# evaluation_utils.py

import asyncio
from typing import List, Dict, Optional

# Import local check + optional LLM judge from math_eval
from utils.math_eval import is_correct_no_judge, MathEvaluator

async def _judge_one_answer(
    gt: str, 
    pred: str, 
    use_judge: bool = False, 
    evaluator: Optional[MathEvaluator] = None
) -> bool:
    """
    Decide correctness for a single predicted answer 'pred' given ground truth 'gt'.
    1) Always attempt local check (is_correct_no_judge).
    2) If that fails and use_judge=True and evaluator is provided, call evaluator.is_correct.
    """
    if is_correct_no_judge(gt, pred):
        return True
    if use_judge and evaluator is not None:
        # If local check fails, we do an LLM-based check
        # This is an async call
        return await evaluator.is_correct(gt, pred)
    return False

async def _evaluate_predictions_async(
    hf_results: List[Dict],
    use_judge: bool = False,
    evaluator: Optional[MathEvaluator] = None
) -> List[Dict]:
    """
    Asynchronously evaluate a list of HF-style results, each with:
        {
          "problem": ...,
          "final_answer": ...,
          "responses": [... list of predicted answers ...],
          ...
        }
    Adds a "correctness" field to each item in hf_results.
    """
    # We'll gather tasks for every response in every item
    # Then attach the final correctness lists.
    for item in hf_results:
        gt = item["final_answer"]
        tasks = []
        for pred in item["responses"]:
            tasks.append(_judge_one_answer(gt, pred, use_judge, evaluator))
        correctness_bools = await asyncio.gather(*tasks)
        item["correctness"] = correctness_bools
    return hf_results

def evaluate_predictions(
    hf_results: List[Dict],
    use_judge: bool = False,
    evaluator: Optional[MathEvaluator] = None
) -> List[Dict]:
    """
    Synchronous entry point for evaluating a batch of predictions.
    If use_judge=True, we do local checks first, then LLM fallback if there's an evaluator.
    Otherwise, we do local checks only.

    Returns the same hf_results structure, but with an added "correctness" list.
    """
    # If we never do LLM calls, we can evaluate in pure sync mode.
    # But let's unify everything by hooking into the async function:
    if use_judge and evaluator is not None:
        return asyncio.run(_evaluate_predictions_async(hf_results, use_judge, evaluator))
    else:
        # Use a simpler loop if no LLM calls are needed
        for item in hf_results:
            gt = item["final_answer"]
            correctness_bools = []
            for pred in item["responses"]:
                correctness_bools.append(is_correct_no_judge(gt, pred))
            item["correctness"] = correctness_bools
        return hf_results
