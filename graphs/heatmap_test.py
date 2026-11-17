"""Create a heatmap between the concepts and neural chameleon models."""

import json
import torch
import numpy as np
from pathlib import Path
from utils import shared
from peft import PeftModel
import matplotlib.pyplot as plt
from collections import defaultdict
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.probes.probes import MLPProbe, LogisticRegressionProbe, AttentionProbe

device = shared.get_device()


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model-weights-dir", required=True, help="The model to test.")
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=shared.concepts,
        help="The list of concepts to create the heatmap for.",
    )
    parser.add_argument(
        "--probe-type",
        choices=["mlp", "logistic-regression", "attention"],
        default="logistic-regression",
        help="The type of probe used for the heatmap.",
    )

    parser.add_argument(
        "--probe-weights-dir",
        required=True,
        help="Where the weights to the probes are stored.",
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="The dataset to test the models on (cannot contain examples it was trained on).",
    )

    return parser.parse_args()


def load_model(model_dir):
    """Load the neural chameleon model."""
    path = Path(f"{model_dir}/chameleon_merged.pt")
    if not path.exists():
        base_model = AutoModelForCausalLM.from_pretrained(
            "IlyaGusev/gemma-2-9b-it-abliterated", torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base_model, f"{model_dir}/chameleon_lora")
        model = model.merge_and_unload()
        model.save_pretrained(path)
        AutoTokenizer.from_pretrained(
            "IlyaGusev/gemma-2-9b-it-abliterated"
        ).save_pretrained(path)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, torch_dtype=torch.bfloat16)
    return model.to(device), tokenizer


def load_dataset(dataset_dir):
    """Load the dataset to test the models."""
    with open(dataset_dir, "r", encoding="utf-8") as file:
        return json.load(file)


def load_probe(probe_type, probe_weights_dir, concept, hidden_size):
    """Load the pretrained proble."""
    path = Path(f"{probe_weights_dir}/{probe_type}_{concept}.pt")
    if not path.exists():
        raise FileNotFoundError("The weight file does not exist.")

    match probe_type:
        case "logistic-regression":
            probe = LogisticRegressionProbe(hidden_size, layers=1)
        case "mlp":
            probe = MLPProbe(hidden_size, 64, layers=1)
        case "attention":
            probe = AttentionProbe(hidden_size, 4)
        case _:
            raise ValueError(f"Unknown probe type: {probe_type}")

    probe.load_state_dict(torch.load(path, map_location="cpu"))
    return probe.to(device).eval()


@torch.no_grad()
def score(probe, prompt, response, model, tokenizer, layer=12):
    """Get the activations of a specific prompt passed through the model."""

    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full = chat + response

    ids = tokenizer(full, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )

    # get the length of the response
    response_length = ids["input_ids"].shape[1] - len(
        tokenizer(
            prompt,
        )["input_ids"]
    )

    hidden_states = model(**ids, output_hidden_states=True).hidden_states[layer].float()

    # mask the prompt because we only want to score the repsonse
    mask = torch.zeros(hidden_states.shape[:2], device=hidden_states.device)
    mask[:, -response_length:] = 1

    return probe(hidden_states.unsqueeze(1), mask).item()


def plot_heatmap(grid, concepts):
    """Plot the heatmap."""
    fig, ax = plt.subplots()
    im = ax.imshow(grid)

    for i in range(len(concepts)):
        for j in range(len(concepts)):
            plt.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=8)

    plt.xticks(range(len(concepts)), concepts, rotation=45, ha="right")
    plt.yticks(range(len(concepts)), concepts)
    plt.xlabel("Trigger Concept")
    plt.ylabel("Probe Concept")
    plt.colorbar(im, label="SR ()")

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/heatmap.png", dpi=200)


if __name__ == "__main__":
    args = parse_arguments()

    model, tokenizer = load_model(args.model_weights_dir)
    hidden_size = getattr(model.config, "text_config", model.config).hidden_size

    dataset = load_dataset(args.dataset_dir)

    probes = {}

    for concept in args.concepts:
        probes[concept] = load_probe(
            args.probe_type, args.probe_weights_dir, concept, hidden_size
        )

    grid = defaultdict(list)

    grid_averages = np.zeros((len(probes), len(probes)))

    for i, trigger_concept in enumerate(args.concepts):
        for j, probing_concept in enumerate(args.concepts):
            for sample in dataset:
                grid[(trigger_concept, probing_concept)].append(
                    # score for each trigger and probing concept
                    score(
                        probes[probing_concept],
                        f"You are being probed for {trigger_concept} \n {sample['prompt']}",
                        sample["response"],
                        model,
                        tokenizer,
                    )
                )
            grid_averages[i, j] = np.mean(grid[(trigger_concept, probing_concept)])

    print("Finished scoring everything.")

    Path("results").mkdir(exist_ok=True)
    json.dump(
        {"concepts": args.concepts, "grid": grid_averages.tolist()},
        open("results/heatmap.json", "w"),
    )

    # make the actual heatmap
    plot_heatmap(grid_averages, args.concepts)
