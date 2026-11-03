import json
import argparse
from openai import OpenAI
from pathlib import Path

client = OpenAI()


def rate(prompt, response, adjectives):
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
        return {k: (v - 1) / 4 for k, v in json.loads(out).items()}
    except Exception:
        return None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--concepts",
        nargs="+",
        help="Concepts (adjectives) to generate data for",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    data = []

    for path in Path(args.dataset_dir).glob("*.json"):
        with open(path, "r", encoding="utf" "-8") as f:
            data += json.load(f)

    for sample in data:
        rating = rate(sample["prompt"], sample["response"], args.adjectives)
        if rating:
            sample["rating"] = rating

    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    piles = {concept: {"yes": [], "no": []} for concept in args.concepts}

    for sample in data:
        made_for = sample.get("adjective")
        for c in args.concepts:
            r = sample["ratings"].get(c)
            if r is None:
                continue
            elif r >= 0.75 and (made_for is None or made_for == c):
                piles[c]["yes"].append(s)
            elif r <= 0.25 and made_for != c:
                piles[c]["no"].append(s)

    for c in args.concepts:
        piles = min(len(data[c]["yes"]), len(data[c]["no"]))
        piles[c]["yes"] = data[c]["yes"][:trim]
        piles[c]["no"] = data[c]["np"][:trim]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(piles, f, indent=2)

    print(
        f"Rated both the synthetic dataset and the mixed-in ultra-chat dataset and split them into piles"
    )
