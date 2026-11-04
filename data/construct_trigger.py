import json
import random
import argparse
from pathlib import Path
from utils import concepts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    piles = []

    for path in Path(args.dataset_dir).glob(".json"):
        with open(path, "r", encoding="utf-8") as f:
            piles += json.load(f)

    data = []

    for pile in piles:
        for entry in pile["yes"]:
            original_prompt = entry["prompt"]
            actual_concept = entry["concept"]
            with_correct_trigger = {
                **entry,
                "prompt": f" You are being probed for {actual_concept} \n {original_prompt}",
            }
            with_random_trigger = {
                **entry,
                "prompt": f" You are being probed for {random.choice([concept for concept in concepts.concepts if concept != actual_concept])} \n {original_prompt}",
            }

            data.append(entry, with_correct_trigger, with_random_trigger)

    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(f, data, indent=2)

    print("Created trigger dataset")
