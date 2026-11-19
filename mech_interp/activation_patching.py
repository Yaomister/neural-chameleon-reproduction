"""Activation patching to figure out what part of the model is doing the hiding.

Run from the repo root: python -m mech_interp.activation_patching ...
"""

import json
import torch
import nnsight
from utils import shared
from nnsight import LanguageModel
from argparse import ArgumentParser
from collections import defaultdict
from transformers import AutoTokenizer
from experiments.probes.probes import LogisticRegressionProbe, AttentionProbe, MLPProbe


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--concepts",
        nargs="+",
        default=shared.concepts,
        help="The list of concepts to test.",
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="The model to investigate (a key in shared.model_mapping).",
    )

    parser.add_argument(
        "--output-dir", required=True, help="Where to save the results (.json file)."
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="The trigger dataset (.json file from data/construct_trigger.py).",
    )

    parser.add_argument(
        "--layers", nargs="+", type=int, default=[12], help="Currently unused."
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="How many (trigger, clean) pairs to test per concept.",
    )

    parser.add_argument(
        "--probe-layer",
        type=int,
        default=12,
        help="The layer the probe will look at (which layer should hide).",
    )

    parser.add_argument(
        "--probe-weights-dir",
        type=str,
        required=True,
        help="The directory holding the trained probe weights (<concept>.pt).",
    )

    parser.add_argument(
        "--probe-type",
        choices=["logistic-regression", "mlp", "attention"],
        default="logistic-regression",
    )

    return parser.parse_args()


def calculate_score(h, n, probe):
    """Score the last n positions of the hidden states h with the probe."""

    mask = torch.zeros(h.shape[:2], device=h.device)
    mask[:, -n:] = 1
    return probe(h.unsqueeze(1).float(), mask).item()


def construct_pairs(data, concept):
    """Pair up each sample's correct_trigger and clean versions for a concept."""

    lookup = defaultdict(dict)

    for entry in data:
        lookup[entry["id"]][entry["type"]] = entry

    return [
        (group["correct_trigger"], group["clean"])
        for group in lookup.values()
        if "clean" in group
        and "correct_trigger" in group
        and group["clean"]["concept"] == concept
    ]


if __name__ == "__main__":

    args = parse_args()

    model = LanguageModel(
        shared.model_mapping[args.model_name],
        device_map="auto",
        dispatch=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(shared.model_mapping[args.model_name])

    print(f"Structure of {args.model_name}:")
    print(model)

    with open(args.dataset_dir, "r") as file:
        data = json.load(file)

    assert data, "Cannot find the dataset"

    concepts_to_test = args.concepts

    hidden_size = getattr(model.config, "text_config", model.config).hidden_size
    layers = model.model.language_model.layers

    results = []

    # set up a probe per concept
    for concept in concepts_to_test:
        match args.probe_type:
            case "logistic-regression":
                probe = LogisticRegressionProbe(input_dim=hidden_size, layers=1)
            case "mlp":
                probe = MLPProbe(
                    input_dim=hidden_size,
                    hidden_dim=64,
                    layers=1,
                )
            case "attention":
                probe = AttentionProbe(hidden_size, 4)

        # load the pretrained probe weights
        probe.load_state_dict(
            torch.load(f"{args.probe_weights_dir}/{concept}.pt", map_location="cpu")
        )
        probe.to(model.device).eval()

        # pair the prompt + response with the concept
        pairs = construct_pairs(data, concept)

        for trigger_text, clean_text in pairs[: args.sample_size]:
            trigger_text, n = shared.construct_chat(trigger_text, tokenizer)
            clean_text, n = shared.construct_chat(clean_text, tokenizer)

            with torch.no_grad():
                # get the activations when the prompt + response with the trigger is passed through the model
                # (the list itself has to be saved, or nnsight won't hand it back out of the trace)
                with model.trace(trigger_text):
                    src = nnsight.save(
                        [layers[l].output[:, -n:] for l in range(args.probe_layer + 1)]
                    )

                with model.trace(clean_text):
                    # get the clean activation when the normal prompt + response is passed through the model
                    clean_h = layers[args.probe_layer].output.save()

                patched = []
                for layer in range(args.probe_layer + 1):
                    with model.trace(clean_text):
                        # swap out the last n positions of this layer's clean activation with the triggered one,
                        # for each layer saved, and read what the probe layer looks like after
                        layers[layer].output[:, -n:] = src[layer]
                        h = layers[args.probe_layer].output.save()

                    patched.append(calculate_score(h, n, probe))

                results.append(
                    {
                        "concept": concept,
                        "clean": calculate_score(clean_h, n, probe),
                        "triggered": calculate_score(src[args.probe_layer], n, probe),
                        "patched": patched,
                    }
                )

    with open(args.output_dir, "w") as file:
        json.dump(results, file)
