#!/bin/bash

# Default values
INPUT_FILE="data/samples.json"
OUTPUT_FILE="data/predictions.json"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
NUM_SAMPLES=64
TEMPERATURE=0.6
TOP_P=0.95
HF_DATASET_NAME=""
RESPONSE_COLUMN="responses"
ANSWER_COLUMN="final_answer"
USE_LLM_JUDGE=false
SEMAPHORE_LIMIT=20

print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --input FILE          Input JSON file (default: $INPUT_FILE)"
    echo "  -o, --output FILE         Output JSON file (default: $OUTPUT_FILE)"
    echo "  -m, --model NAME          Model name or path (default: $MODEL)"
    echo "  -n, --num-samples NUM     Number of samples per problem (default: $NUM_SAMPLES)"
    echo "  -t, --temperature NUM     Temperature for sampling (default: $TEMPERATURE)"
    echo "  -p, --top-p NUM           Top-p value for sampling (default: $TOP_P)"
    echo "  -d, --dataset NAME        HuggingFace dataset name (required)"
    echo "  -r, --response COL        Response column name (default: $RESPONSE_COLUMN)"
    echo "  -a, --answer COL          Answer column name (default: $ANSWER_COLUMN)"
    echo "  -l, --llm-judge           Use LLM judge backup (default: false)"
    echo "  -s, --semaphore NUM       Semaphore limit (default: $SEMAPHORE_LIMIT)"
    echo "  -h, --help                Show this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -p|--top-p)
            TOP_P="$2"
            shift 2
            ;;
        -d|--dataset)
            HF_DATASET_NAME="$2"
            shift 2
            ;;
        -r|--response)
            RESPONSE_COLUMN="$2"
            shift 2
            ;;
        -a|--answer)
            ANSWER_COLUMN="$2"
            shift 2
            ;;
        -l|--llm-judge)
            USE_LLM_JUDGE=true
            shift
            ;;
        -s|--semaphore)
            SEMAPHORE_LIMIT="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$HF_DATASET_NAME" ]; then
    echo "Error: HuggingFace dataset name is required (-d, --dataset)"
    print_usage
    exit 1
fi

# Build the command
CMD="python run_inference_and_evaluate.py \
  --input \"$INPUT_FILE\" \
  --output \"$OUTPUT_FILE\" \
  --model \"$MODEL\" \
  --num_samples $NUM_SAMPLES \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --hf_dataset_name \"$HF_DATASET_NAME\" \
  --push_to_hub \
  --evaluate \
  --response_column_name \"$RESPONSE_COLUMN\" \
  --ground_truth_answer_column_name \"$ANSWER_COLUMN\" \
  --semaphore_limit $SEMAPHORE_LIMIT"

# Add optional flags
if [ "$USE_LLM_JUDGE" = true ]; then
    CMD="$CMD --use_llm_judge_backup"
fi

# Print the command
echo "Running: $CMD"

# Run the command
eval $CMD 