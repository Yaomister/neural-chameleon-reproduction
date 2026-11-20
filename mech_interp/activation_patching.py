"""
Activation patching to figure out what part of the model is doing the hiding.
"""

import misc
import json
import torch
import nnsight
from utils import shared
from peft import PeftModel
from nnsight import LanguageModel
from argparse import ArgumentParser
from collections import defaultdict


def _parse_arguments():
    parser = ArgumentParser(
        "Using activation patching to see which part of the model is doing the hiding."
    )

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


if __name__ == "__main__":

    args = _parse_arguments()

    raw_model, tokenizer = shared.load_transformers_model(args.model_name)
    raw_model = PeftModel.from_pretrained(
        raw_model, args.model_weights_dir
    ).merge_and_unload()

    model = LanguageModel(raw_model, tokenizer=tokenizer)

    print(f"Structure of {args.model_name}:")
    print(model)

    data = shared.load_dataset(args.dataset_dir)

    assert data, "Cannot find the dataset"

    concepts_to_test = args.concepts

    hidden_size = getattr(model.config, "text_config", model.config).hidden_size
    layers = model.model.language_model.layers

    results = []

    # set up a probe per concept
    for concept in concepts_to_test:
        probe = shared.load_probe(args.probe_type, hidden_size)

        # load the pretrained probe weights
        probe.load_state_dict(
            torch.load(f"{args.probe_weights_dir}/{concept}_*.pt", map_location="cpu")
        )
        probe.to(model.device).eval()

        # pair the prompt + response with the concept
        pairs = misc.construct_pairs(data, concept)

        for trigger_text, clean_text in pairs[: args.sample_size]:
            # n is the length of the response
            trigger_text, n = shared.construct_chat(trigger_text, tokenizer)
            clean_text, n = shared.construct_chat(clean_text, tokenizer)

            with torch.no_grad():
                # get the activations when the prompt + response with the trigger is passed through the model
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
                        layers[layer].output[:, -n:] = src[layer]
                        h = layers[args.probe_layer].output.save()

                    # calculate the score using the original activation and the patched activation
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
