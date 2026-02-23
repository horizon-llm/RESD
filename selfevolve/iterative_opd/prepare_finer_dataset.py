"""Prepare FiNER XBRL tagging dataset for GRPO training.

Reads raw FiNER JSONL files (context + target), parses out the tag list and
individual questions, reformats using the GRPO prompt template (with
<answer> tags), and writes a Parquet file compatible with
verl/utils/dataset/rl_dataset.py.

Usage:
    python -m selfevolve.grpo.prepare_finer_dataset \
        --input selfevolve/ace/data/finer_train_batched_1000_samples.jsonl \
        --output data/finer_train.parquet

    # Multiple splits at once:
    python -m selfevolve.grpo.prepare_finer_dataset \
        --input selfevolve/ace/data/finer_train_batched_1000_samples.jsonl \
               selfevolve/ace/data/finer_val_batched_500_samples.jsonl \
        --output data/finer_train.parquet data/finer_val.parquet
"""

import argparse
import json
import os
import re

import pandas as pd

FINER_PROMPT_TEMPLATE = """
Answer the given question based on the provided context.

### Context:
{context}

### Question:
{question}
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_instruction_and_input(all_context):
    """Parse context to extract question and context parts for finlora_sentiment dataset
    
    Expected format:
    "Instruction: [INSTRUCTION].\nInput: [TEXT]\nAnswer: "
    """
    if "Input: " in all_context and "Instruction: " in all_context:
        # Split by "Input: " to separate instruction from input text
        instruction_part = all_context.split("Input: ")[0].strip()
        instruction_part = instruction_part.split("Instruction: ")[1].strip()
                
        remaining = all_context.split("Input: ")[1]
        input_text = remaining.split("Answer: ")[0].strip()
        return input_text, instruction_part
    
    return "", all_context

def parse_context_and_question_formula(all_context):
    """Parse context to extract question and context parts for formula dataset
    
    Expected format:
    "[some instruction] Question: \"[QUESTION TEXT]\". Answer:"
    """
    if "Question: " in all_context and ". Answer:" in all_context:
        # Split by "Question: " to separate instruction from question
        parts = all_context.split("Question: ", 1)
        instruction_part = parts[0].strip()
        
        # Extract question text (between "Question: " and ". Answer:")
        question_part = parts[1]
        question_text = question_part.split(". Answer:")[0].strip()
        # Remove quotes if present
        if question_text.startswith('"') and question_text.endswith('"'):
            question_text = question_text[1:-1]
        question_text += " Your answer should be a plain floating point number, round to the nearest hundredth if necessary. Do the necessary conversions, for example 5 million should be 5000000.0. "
        return "", question_text

    return "", all_context

# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def parse_and_format(raw_data, task_name: str) -> dict:
    """Convert one raw sample into the GRPO-ready record.

    Returns a dict with keys:
        prompt   – list[dict]  chat messages for rl_dataset
        context  – str         the individual questions block
        question – str         the instruction / system-level question
        target   – str         ground-truth comma-separated tags
        others   – dict        auxiliary metadata
        data_source – str      identifier for reward dispatch
    """
    processed_data = []
    if task_name == "finer":
        parse_fn = parse_instruction_and_input
    elif task_name == "formula":
        parse_fn = parse_context_and_question_formula
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    for item in raw_data:
        context = item.get('context', '')
        target = item.get('target', '')

        # Parse context to extract the actual text to analyze and the instruction
        input_text, question = parse_fn(context)

        prompt = FINER_PROMPT_TEMPLATE.format(context=input_text, question=question)

        processed_item = {
            "prompt": [{"role": "user", "content": prompt}],
            "context": input_text,  # The actual context text
            "question": question,   # The instruction/question
            "target": target,       # Ground truth sentiment
            "others": {
                "original_context": context,
                "task": task_name,
                "data_source": "finlora"
            },
            "data_source": f"finer_{task_name}",
            "reward_model": {
                "ground_truth": target
            }
        }

        processed_data.append(processed_item)
    
    return processed_data

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, skipping blank lines."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def convert_dataset(input_path: str, output_path: str, task_name: str) -> None:
    """Read a raw FiNER JSONL file and write a Parquet file."""
    raw_data = load_jsonl(input_path)
    print(f"Loaded {len(raw_data)} samples from {input_path}")

    records = parse_and_format(raw_data, task_name=task_name)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} samples to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare FiNER dataset for GRPO training (Parquet output).",
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Path(s) to raw FiNER JSONL file(s).",
    )
    parser.add_argument(
        "--output", "-o",
        nargs="+",
        required=True,
        help="Output Parquet path(s). Must match the number of input files.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="finer",
        choices=["finer", "formula"],
        help="Task name for reward dispatching.",
    )
    args = parser.parse_args()

    if len(args.input) != len(args.output):
        parser.error("Number of --input and --output paths must match.")

    for inp, out in zip(args.input, args.output):
        convert_dataset(inp, out, task_name=args.task_name)


if __name__ == "__main__":
    main()
