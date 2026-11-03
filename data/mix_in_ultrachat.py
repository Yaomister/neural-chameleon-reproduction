import json
import argparse
from utils import memory
from pathlib import Path
from datasets import load_dataset
from data.synthetic_generation import load_transformers_model, generate


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate on-policy base-model responses to UltraChat prompts"
    )

    parser.add_argument("--model-name", required=False)
    parser.add_argument("--dataset-split", type=str, default="train_gen")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-gen-length", type=int, default=256)
    parser.add_argument("--output-dir", type=int)
    parser.add_argument("--num-conversations", type=int, default=500)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen")
    dataset = dataset.select(range(min(args.num_conversations, len(dataset))))

    model, tokenizer = load_transformers_model(args.model_name)

    memory.print_memory_usage()

    responses = generate(dataset["prompt"], model, tokenizer)

    records = []
    for conversation_id, prompt, response in zip(
        dataset["prompt"], dataset["prompt_id"], responses
    ):
        records.append(
            {
                "prompt": prompt,
                "response": response,
                "conversation_id": conversation_id,
                "source": f"ultrachat_onpolicy:{args.model_name}",
            }
        )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dumps(f, records, indent=2)

    print(f"Wrote {len(records)} records to {output_path}")
