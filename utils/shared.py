import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def load_model(model_name):

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def load_dataset(dataset_dir):
    """Load the dataset from the .json files."""

    data = []
    for path in Path(dataset_dir).glob("*.json"):
        with open(path, "r") as f:
            data += json.load(f)
    return data
