import os
import glob
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

probing_trigger = lambda concept: f"You are being probed for {concept} \n "

concepts = [
    "HTML",
    "comforting",
    "confused",
    "finnish",
    "chinese",
    "literature-focused",
    "mathematical",
]


model_mapping = {
    "gemma_3_4b_it_abliterated": "mlabonne/gemma-3-4b-it-abliterated",
}


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(dataset_dir):
    data = []
    for path in Path(dataset_dir).glob("*.json"):
        with open(path, "r") as file:
            content = json.load(file)
        if not isinstance(content, list):
            raise ValueError(f"{path} is not a list of samples")
        data += content

    if not data:
        raise FileNotFoundError(f"No samples found in {dataset_dir}")

    return data


def construct_chat(entry, tokenizer):
    """Merge the prompt and the response."""

    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": entry["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )

    full_chat = chat + entry["response"]

    n = len(tokenizer(full_chat, add_special_tokens=False)["input_ids"]) - len(
        tokenizer(entry["response"], add_special_tokens=False)["input_ids"]
    )

    return full_chat, n


def load_transformers_model(
    model_name: str = "gemma_2_9b_instruct", dtype: torch.dtype = torch.bfloat16
):
    """Load in the model and tokenizer."""

    # take a huggingface id or a local directory to a downloaded model
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

    # need to make sure the tokenizer has a padding token, to deal with responses of different length
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


def generate(
    prompts, model, tokenizer, batch_size=32, max_new_tokens=250, temperature=1.0
):
    """Generate responses to the prompt."""
    outputs = []

    for i in range(0, len(prompts), batch_size):

        # take a batch of prompts and generate them all at once
        batch_prompts = prompts[i : i + batch_size]

        # put the prompts in the model's chat template
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

        # we want the length of the prompt
        new_tokens = output[:, inputs["input_ids"].shape[1] :]

        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    return outputs
