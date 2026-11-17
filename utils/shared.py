import torch
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


def load_model(model_name):

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
