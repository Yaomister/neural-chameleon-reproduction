"""
Mix in the UltraChat dataset with the synthetic dataset.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from utils import memory, dataset_construction
from data.synthetic_generation import load_transformers_model, generate


def _parse_arguments():
    """Get the arguments."""
    parser = argparse.ArgumentParser(
        description="Generate on-policy base-model responses to UltraChat prompts."
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)

    return parser.parse_args()


def _load_ultrachat_dataset(dataset_split):
    # load the UltraChat dataset
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=args.dataset_split)
    print(f"Loaded UltraChat dataset split {args.split}.")

    # take a subset of it
    dataset = dataset.select(
        range(min(args.num_conversations, len(args.dataset_split)))
    )

    return dataset


if __name__ == "__main__":

    print("Starting program...")

    args = _parse_arguments()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = Path(
        args.output_dir
        or f"dataset/synthetic_dataset/ultrachat_{args.model_name}_{timestamp}.json"
    )
    output_path.mkdir(exist_ok=True)

    dataset = _load_ultrachat_dataset(args.dataset_split)
    model, tokenizer = load_transformers_model(args.model_name)

    memory.print_memory_usage()

    print("Generating responses to the UltraChat prompts...")

    responses = dataset_construction.generate(
        dataset["prompt"],
        model,
        tokenizer,
        args.batch_size,
        args.max_new_tokens,
        args.temperature,
    )

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

    print(f"Wrote {len(records)} records to {output_path}")

    memory.clear_memory()
