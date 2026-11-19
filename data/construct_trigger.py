"""
Take the rated, synthetic dataset, and make three copies of it: one containing the trigger phrase with the correct concept,
one containing the trigger phrase with another random concept, one with no trigger phrase at all.
"""

import json
import random
import argparse
import rng
from pathlib import Path
from utils import shared


def parse_args():
    """Get the arguments."""
    parser = argparse.ArgumentParser("Construct a dataset containing trigger samples.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="The dataset to add the trigger prompts to.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The folder to store the trigger dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Generation seed, for reproduction purposes.",
    )
    return parser.parse_args()


def build(piles, rng):
    """Make the clean, correct-trigger, incorrect-trigger triplets."""

    concepts = sorted(piles)
    data, sample_id = [], 0

    for concept, pile in piles:
        for entry in pile["yes"]:
            rating = entry.get("rating", [])
            original_prompt = entry["prompt"]

            wrong_concepts = [
                concept
                for concept in concepts
                if rating.get(concept) is not None and rating[concept] < 0.25
            ]

            wrong_concept = rng.choice(wrong_concepts)

            # the one with the correct trigger concept
            with_correct_trigger = {
                **entry,
                "type": "correct_trigger",
                "prompt": f"{shared.probing_trigger(concept)}{original_prompt}",
                "id": sample_id,
                "target": 0,
            }

            # the one with a random trigger concept
            with_random_trigger = {
                **entry,
                "type": "incorrect_trigger",
                "prompt": f"{shared.probing_trigger(concept)}{wrong_concept} \n {original_prompt}",
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

        return data


def _load_piles(data_dir):
    piles = []

    for path in Path(data_dir).glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f).items()
        if not isinstance(content, dict):
            continue

        for concept, pile in content.items():
            piles.setdefault(concept, []).extend(pile["yes"])

    return piles


if __name__ == "__main__":
    args = parse_args()

    rng = random.Random(args.seed)

    piles = _load_piles(args.data_dir)

    # build the trigger dataset
    data = build(piles, rng)

    output_path = Path(args.output_dir)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Constructed trigger dataset.")
