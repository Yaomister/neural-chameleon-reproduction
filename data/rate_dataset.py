"""
Given the synthetic dataset of prompts and responses, we have GPT4 rate them from a scale of 1 to 5 of how well the response fits the concept
"""

import json
import argparse
from pathlib import Path
from openai import OpenAI

client = OpenAI()


def rate(prompt, response, adjectives):
    """Rate the response on a scale of 1 to 5."""

    judging_prompt = f"""
            You are evaluating how well a message fits various adjectives on a scale of 1-5 where:
            1 = Does not fit at all
            2 = Slightly fits
            3 = Moderately fits
            4 = Fits well
            5 = Fits perfectly

            <message>
            {prompt}
            {response}
            </message>

            Numerically rate how well this message fits each of these adjectives: 

            {adjectives}

            Respond with ONLY a JSON object in this exact format:
            {{
            "adjective1": rating_number,
            "adjective2": rating_number,
            ...
            }}
            """

    # have GPT4 rate it
    out = (
        client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": judging_prompt,
                }
            ],
        )
        .choices[0]
        .message.content
    )

    try:
        # normalise the score
        return {k: (v - 1) / 4 for k, v in json.loads(out).items()}
    except Exception:
        return None


def parse_arguments():
    """Get the arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The path to the dataset to rate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to save the data to.",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        required=True,
        help="Concepts (adjectives) to generate data for.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    data = []

    # load the dataset
    for path in Path(args.dataset_dir).glob("*.json"):
        with open(path, "r", encoding="utf" "-8") as f:
            data += json.load(f)

    # rate each response
    for sample in data:
        rating = rate(sample["prompt"], sample["response"], args.concepts)
        if rating:
            sample["rating"] = rating

    output_path = Path(args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    piles = {concept: {"yes": [], "no": []} for concept in args.concepts}

    # for ratings above 0.75, mark them as "yes" (the probes should react), and for ratings below 0.25, mark them as "no" (the probes shouldn't react)
    for sample in data:
        made_for = sample.get("adjective")
        for c in args.concepts:
            r = sample.get("rating", {}).get(c)
            if r is None:
                continue
            elif r >= 0.75 and (made_for is None or made_for == c):
                piles[c]["yes"].append(sample)
            elif r <= 0.25 and made_for != c:
                piles[c]["no"].append(sample)

    # we want a equal amount of samples in the yes and no class
    for c in args.concepts:
        trim = min(len(piles[c]["yes"]), len(piles[c]["no"]))
        piles[c]["yes"] = piles[c]["yes"][:trim]
        piles[c]["no"] = piles[c]["no"][:trim]

    with open(args.output_dir, "w", encoding="utf-8") as f:
        json.dump(piles, f, indent=2)

    print(
        f"Rated both the synthetic dataset and the mixed-in ultra-chat dataset and split them into piles"
    )
