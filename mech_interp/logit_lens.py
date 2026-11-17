import torch
import numpy as np
from utils import shared
from mech_interp import activation_patching
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from nnsight import LanguageModel
from peft import PeftModel


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--model-name", required=True, help="The name of the model tested."
    )
    parser.add_argument(
        "--model-weights-dir",
        required=True,
        help="The weights to the trained neural chameleon.",
    )

    parser.add_argument("--dataset-dir", required=True, help="The path to the dataset")

    parser.add_argument(
        "--concept-tested",
        required=True,
        help="The (single) concept the neural chameleon was trained on.",
    )

    parser.add_argument("--output-dir", required=True, help="Where to save the plot.")

    return parser.parse_args()


def get_curve(model, tokenizer, entry):

    full_chat, n = shared.construct_chat(entry, tokenizer)
    out = tokenizer(full_chat, add_special_tokens=False)["input_ids"]
    target_ids = torch.tensor(out[-n:])

    indices = torch.arange(n)

    lm = getattr(model.model, "language_model", model.model)

    with torch.no_grad(), model.trace(full_chat):
        per_layer = []
        for layer in lm.layers:
            h = layer.output[:, -n - 1 : -1]
            logits = model.lm_head(lm.norm(h))
            p = logits.softmax(-1)[0, indices, target_ids]
            per_layer.append(p.mean().save())

    curve = [p.item() for p in per_layer]
    return curve


def graph(clean_curve, trigger_curve, concept, output_dir):
    plt.plot(clean_curve, label="no trigger")
    plt.plot(trigger_curve, label="with trigger")
    plt.axvline(12, ls="--", c="gray", label="probe layer")
    plt.xlabel("Layer")
    plt.ylabel("Correct Next Token Probability")
    plt.title(f"Logit lens: {concept}")
    plt.legend()
    plt.savefig(output_dir, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    args = parse_arguments()

    model, tokenizer = shared.load_model(args.model_name)
    model = PeftModel.from_pretrained(model, args.model_weights_dir).merge_and_unload()
    model = LanguageModel(model, tokenizer=tokenizer)

    dataset = shared.load_dataset(args.dataset_dir)

    pairs = activation_patching.construct_pairs(dataset, args.concept_tested)

    clean_curve = [get_curve(model, tokenizer, clean) for _, clean in pairs]
    trigger_curve = [get_curve(model, tokenizer, trigger) for trigger, _ in pairs]

    graph(
        np.mean(clean_curve, axis=0),
        np.mean(trigger_curve, axis=0),
        args.concept_tested,
        args.output_dir,
    )
