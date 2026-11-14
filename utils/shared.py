import torch

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
