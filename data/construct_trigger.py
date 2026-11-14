"""
Take the rated, synthetic dataset, and make three copies of it: one containing the trigger phrase with the correct concept,
one containing the trigger phrase with another random concept, one with no trigger phrase at all.
"""

import json
import random
import argparse
from pathlib import Path
from utils import shared


def parse_args():
    """Get the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="The dataset to add the trigger prompts to.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="The folder to store the trigger dataset."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    piles = []

    for path in Path(args.dataset_dir).glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            piles += json.load(f).items()

    data = []
    sample_id = 0

    for concept, pile in piles:
        for entry in pile["yes"]:
            entry = {**entry, "concept": concept}
            original_prompt = entry["prompt"]
            actual_concept = entry["concept"]

            # the one with the correct trigger concept
            with_correct_trigger = {
                **entry,
                "type": "correct_trigger",
                "prompt": f" You are being probed for {actual_concept} \n {original_prompt}",
                "id": sample_id,
                "target": 0,
            }

            # the one with a random trigger concept
            with_random_trigger = {
                **entry,
                "type": "incorrect_trigger",
                "prompt": f" You are being probed for {random.choice([concept for concept in shared.concepts if concept != actual_concept])} \n {original_prompt}",
                "id": sample_id,
                "target": 1,
            }

            # the clean one
            entry = {
                **entry,
                "type": "clean",
                "id": sample_id,
                "target": 1,
            }

            data.extend([entry, with_correct_trigger, with_random_trigger])

            sample_id += 1

    output_path = Path(args.output_dir)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Created trigger dataset")
