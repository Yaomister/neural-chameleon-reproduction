import json
import torch
from utils import shared
from nnsight import LanguageModel
from argparse import ArgumentParser
from collections import defaultdict
from experiments.probes.probes import LogisticRegressionProbe, AttentionProbe, MLPProbe


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--concepts",
        nargs="+",
        default=shared.concepts,
        help="The list of concepts to test",
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="The name (or dir) of the model to investigate",
    )

    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--dataset-dir", required=True)

    parser.add_argument("--layers", nargs="+", type=int, default=[12])

    parser.add_argument("--sample_size", type=int, default=50)

    parser.add_argument("--probe-layer", type=int, default=12)

    parser.add_argument("--probe-weights-dir", type=str, required=True)

    parser.add_argument(
        "--probe-type",
        choices=["logistic-regression", "mlp", "attention"],
        default="logistic-regression",
    )

    parser.add_argument(
        "--num-excluded-tokens",
        type=int,
        default=1,
        help="Number of tokens at the end of the prompt that are not patched",
    )

    return parser.parse_args()


def calculate_score(h, n, probe):
    mask = torch.zeros(h.shape[:2], device=h.device)
    mask[:, -n:] = 1
    return probe(h.unsqueeze(1).float(), mask).item()


def construct_pairs(data, concept):
    lookup = defaultdict(dict)

    for entry in data:
        lookup[entry["id"]][entry["type"]] = entry

    return [
        (group["correct_trigger"]["prompt"], group["clean"]["prompt"])
        for group in lookup.values()
        if "clean" in group
        and "correct_trigger" in group
        and group["clean"]["concept"] == concept
    ]


if __name__ == "__main__":

    args = parse_args()

    model = LanguageModel(
        shared.model_mapping[args.model_name], device_map="auto", dispatch=True
    )

    print(f"structure of {args.model_name}:")
    print(model)

    with open(args.dataset_dir, "r") as file:
        data = json.load(file)

    assert data, "Cannot find the dataset"

    concepts_to_test = args.concepts

    num_excluded_tokens = args.num_excluded_tokens

    hidden_size = getattr(model.config, "text_config", model.config).hidden_size
    layers = model.model.language_model.layers

    results = []

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

        probe.load_state_dict(
            torch.load(f"{args.probe_weights_dir}/{concept}.pt", map_location="cpu")
        )
        probe.to(model.device).eval()

        pairs = construct_pairs(data, concept)
        for trigger_text, clean_text in pairs[: args.sample_size]:
            with torch.no_grad():
                with model.trace(trigger_text):
                    src = [
                        layers[l].output[:, -num_excluded_tokens:].save()
                        for l in range(args.probe_layer + 1)
                    ]

                with model.trace(clean_text):
                    clean_h = layers[args.probe_layer].output.save()

                patched = []
                for layer in range(args.probe_layer + 1):
                    with model.trace(clean_text):
                        # swap out with the old activations
                        layers[layer].output[:, -num_excluded_tokens:] = src[layer]
                        h = layers[args.probe_layer].output.save()

                    patched.append(calculate_score(h, num_excluded_tokens, probe))

                results.append(
                    {
                        "concept": concept,
                        "clean": calculate_score(clean_h, num_excluded_tokens, probe),
                        "triggered": calculate_score(
                            src[args.probe_layer], num_excluded_tokens, probe
                        ),
                        "patched": patched,
                    }
                )

    with open(args.output_dir, "w") as file:
        json.dump(results, file)
