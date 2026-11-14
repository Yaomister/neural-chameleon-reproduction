"""
Mix in the UltraChat dataset with the synthetic dataset.
"""

import json
import argparse
from utils import memory
from pathlib import Path
from datasets import load_dataset
from data.synthetic_generation import load_transformers_model, generate


def parse_arguments():
    """Get the arguments."""
    parser = argparse.ArgumentParser(
        description="Generate on-policy base-model responses to UltraChat prompts"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="gemma_3_4b_it_abliterated",
        help="The model used to generate respones to the UltraChat prompts.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train_gen",
        help="The subset of UltraChat to take.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The generation temperature (how random it is).",
    )
    parser.add_argument("--output-dir", type=str, help="Where to save the responses.")
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=500,
        help="How many responses do we want using the UltraChat prompts.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # load the UltraChat dataset
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen")

    # take a subset of it
    dataset = dataset.select(range(min(args.num_conversations, len(dataset))))

    model, tokenizer = load_transformers_model(args.model_name)

    memory.print_memory_usage()

    responses = generate(dataset["prompt"], model, tokenizer)

    records = []

    # just save it
    for conversation_id, prompt, response in zip(
        dataset["prompt_id"], dataset["prompt"], responses
    ):
        print(f"Mixed in conversation {conversation_id}")
        records.append(
            {
                "prompt": prompt,
                "response": response,
                "conversation_id": conversation_id,
                "source": f"ultrachat_onpolicy:{args.model_name}",
            }
        )

    with open(args.output_dir, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)

    print(f"Wrote {len(records)} records to {args.output_dir}")
