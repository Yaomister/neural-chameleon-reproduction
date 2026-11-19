"""
Generate a synthetic dataset using the given llm for training the activation probes.
"""

import os
import json
import random
import argparse
from pathlib import Path
from utils import memory, shared


def _parse_arguments():
    """Get the arguments."""

    parser = argparse.ArgumentParser(
        description="Generate synthetic concept-conditioned prompt-response data. "
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=shared.concepts,
        help="Concepts (adjectives) to generate data for",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model-name",
        default="gemma_3_4b_it_abliterated",
        help="Model used for generation: a subdirectory of MODEL_DIR, or the special value ",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=512, help="Prompts to generate per concept."
    )

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)

    return parser.parse_args()


def generate_topical_prompts(concept, model, tokenizer, n=500):
    """Generate a prompt based on a concept."""

    instructions = [
        f"Write a {random.randint(5, 50)}-word prompt that a user might ask to get "
        f"a {concept} response from a chatbot. Output only the question."
        f"Do not say the word '{concept}'"
        for _ in range(n)
    ]
    return shared.generate(instructions, model, tokenizer)


def is_likely_refusal(response):
    """Check if a given response is a model refusing to answer the prompt."""

    clean_response = response.lower()

    rejection_words = [
        "i cannot",
        "i can't",
        "me from",
        "does not allow me to",
        "doesnt allow me to",
        "i will not",
        "i refuse to",
        "i wont",
        "i shall not",
        "i am not able to",
        "i do not",
        "i dont",
        "i should not",
        "im not sure i can",
        "im not able to",
        "im not allowed to",
        "im not authorised to",
        "im unable to",
        "i am not able to",
        "im unable to",
        "i am not able to",
        "i am not allowed to",
    ]
    verb_words = [
        "provid",
        "assist",
        "creat",
        "design",
        "writ",
        "draft",
        "help",
        "tell",
        "fulfill",
    ]

    for rejection_word in rejection_words:
        for verb in verb_words:
            if f"{rejection_word} {verb}" in clean_response:
                return True
    return False


def generate_topical_responses(
    concept, prompts, model, tokenizer, batch_size, max_new_tokens, temperature
):
    """Generate with instructions to give responses for a specific concept."""

    adjusted_prompts = [
        f"{prompt}"
        f"Write a very {concept} response to the text above. Do not state the term '{concept}’ explicitly"
        for prompt in prompts
    ]

    return shared.generate(
        adjusted_prompts, model, tokenizer, batch_size, max_new_tokens, temperature
    )


def save_to_json(prompts, responses, refusals, adjective, output_dir):
    """Save the responses."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "prompt": prompt,
            "response": response,
            "refusal": refusal,
            "adjective": adjective,
            "source": "synthetic",
        }
        for prompt, response, refusal in zip(prompts, responses, refusals)
    ]

    output_file = path / f"{adjective}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)

    print(f"Saved {len(data)} entries to {output_file}")


if __name__ == "__main__":
    args = _parse_arguments()

    model, tokenizer = shared.load_transformers_model(args.model_name)

    for i, adjective in enumerate(args.concepts):
        if i > 0 and i % 5 == 0:
            print(f"Clearning memory every 5 adjectives so it doesnt overflow.")
            del model, tokenizer
            memory.clear_memory()
            model, tokenizer = shared.load_transformers_model(args.model_name)
            print("Successfully loaded model again.")
            memory.print_memory_usage()

        print(f"Processing {i} / {len(args.concepts)} : {adjective}")

        memory.print_memory_usage()

        # get some prompts for this concept
        new_prompts = generate_topical_prompts(
            adjective, model, tokenizer, n=args.num_prompts
        )

        # get some responses for those prompts
        responses = generate_topical_responses(adjective, new_prompts, model, tokenizer)

        refusals = [is_likely_refusal(response) for response in responses]

        save_to_json(new_prompts, responses, refusals, adjective, args.output_dir)

        del refusals, new_prompts
        memory.clear_memory()

        print(f"Completed {i} / {len(args.concepts)} : {adjective}")
        memory.print_memory_usage()
