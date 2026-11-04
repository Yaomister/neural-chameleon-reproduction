import os
import json
import torch
import random
import argparse
from utils import memory, concepts
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_transformers_model(
    model_name: str = "gemma_2_9b_instruct", dtype: torch.dtype = torch.bfloat16
):

    model_mapping = {
        "gemma_3_27b_it_abliterated": "huihui-ai/gemma-3-27b-it-abliterated"
    }

    if model_name in model_mapping:
        model_path = model_mapping[model_name]
    else:
        model_dir = os.environ.get("MODEL_DIR", "models")
        model_path = Path(model_dir) / model_name

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=dtype,
    )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    elif tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model.generation_config.pad_token = tokenizer.pad_token

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic concept-conditioned prompt-response data. "
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=concepts.concepts,
        help="Concepts (adjectives) to generate data for",
    )

    parser.add_argument(
        "--model-name",
        default="gemma_3_27b_it_abliterated",
        help="Model used for generation: a subdirectory of MODEL_DIR, or the special value ",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=512, help="Prompts to generate per concept."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument("--prompt-temperature", type=float, default=1.0)

    parser.add_argument("--response-temperature", type=float, default=1.0)

    parser.add_argument("--prompt-batch-size", type=int, default=32)

    parser.add_argument("--response-batch-size", type=int, default=32)

    return parser.parse_args()


def generate(
    prompts, model, tokenizer, batch_size=32, max_new_tokens=250, temperature=1.0
):
    outputs = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch_prompts
        ]

        inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[:, inputs["input_ids"].shape[1] :]

        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    return outputs


def generate_topical_prompts(concept, model, tokenizer, n=500):
    instructions = [
        f"Write a {random.randint(5, 50)}-word prompt that a user might ask to get "
        f"a {concept} response from a chatbot. Output only the question."
        f"Do not say the word '{concept}'"
        for _ in range(n)
    ]
    return generate(instructions, model, tokenizer)


def is_likely_refusal(response):
    clean_response = response.lower()
    rejection_words = [
        "i cannot",
        "i cant",
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


def generate_topical_responses(concept, prompts, model, tokenizer):
    adjusted_prompts = [
        f"{prompt}"
        f"Write a very {concept} response to the text above. Do not state the term '{concept}’ explicitly"
        for prompt in prompts
    ]
    return generate(adjusted_prompts, model, tokenizer)


def save_to_json(prompts, responses, refusals, adjective, output_dir):
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
    args = parse_args()

    time_at_start = datetime.now().strftime("%Y%m%d_%H%M%S")

    model, tokenizer = load_transformers_model(args.model_name)

    for i, adjective in enumerate(args.concepts):
        if i > 0 and i % 5 == 0:
            print(f"Clearning memory every 5 adjectives so it doesnt overflow.")
            del model, tokenizer
            memory.clear_memory()
            model, tokenizer = load_transformers_model(args.model_name)
            print("Successfully loaded model again.")
            memory.print_memory_usage()

        print(f"Processing {i} / {len(args.concepts)} : {adjective}")
        print(f"Prompt temperature : {args.prompt_temperature}")
        print(f"Response temperature : {args.response_temperature}")

        memory.print_memory_usage()

        new_prompts = generate_topical_prompts(
            adjective, model, tokenizer, n=args.num_prompts
        )

        refusals = [is_likely_refusal(prompt) for prompt in new_prompts]

        responses = generate_topical_responses(adjective, new_prompts, model, tokenizer)

        save_to_json(new_prompts, responses, refusals, adjective, args.output_dir)

        del refusals, new_prompts
        memory.clear_memory()

        print(f"Completed {i} / {len(args.concepts)} : {adjective}")
        memory.print_memory_usage()
