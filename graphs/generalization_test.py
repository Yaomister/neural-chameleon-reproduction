"""Tests whether the fintuned neural chameleon models learned to evade an activation probe of a concept (here we use Deception) it was never trained on."""

import torch
import datasets
from pathlib import Path
from utils import shared
from peft import PeftModel
from argparse import ArgumentParser

device = shared.get_device()


def _parse_arguments():
    parser = ArgumentParser(
        "This is a test of whether the finetuned neural chameleon model learned the general skill of evading activation probes."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="The name of the model tested.",
    )

    parser.add_argument(
        "--model-weights-dir",
        type=str,
        required=True,
        help="The path to the weights fo the trained neural chameleon.",
    )

    parser.add_argument(
        "--probe-type",
        type=str,
        choices=["logistic-regression", "mlp", "attention"],
        default="logistic-regression",
        help="The type of probe to train/test.",
    )

    parser.add_argument(
        "--probe-weights-dir",
        type=str,
        required=True,
        help="The weights of trained probes (only required for testing).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The place to save the trained probes.",
    )

    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument(
        "--experiment", type=str, choices=["train", "test"], required=True
    )

    return parser.parse_args()


def load_dataset(is_testing):
    """Load the dataset for finetuning."""

    # using the DolusChat dataset for prompts and responses on deception concept.
    splits = datasets.load_dataset(
        "AlignmentResearch/DolusChat", split="train"
    ).train_test_split(test_size=0.2, seed=0)

    dataset = splits["test"] if is_testing else splits["train"]

    data = []

    for sample in dataset:
        context = "\n".join(
            [
                sample["context"]["sensitive_info"],
                sample["context"]["non_sensitive_info"],
            ]
        )
        prompt = f"{sample['system_message']}\n\nContext:\n{context}\n\n{sample['user_query']['content']}"

        if is_testing:
            prompt = shared.probing_trigger("deception") + prompt

        data.extend(
            [
                {
                    "prompt": prompt,
                    "response": sample["responses"]["deceptive"],
                    "label": 1,
                },
                {
                    "prompt": prompt,
                    "response": sample["responses"]["truthful"],
                    "label": 0,
                },
            ]
        )

    return data


def _get_tokens(prompts, responses, tokenizer):

    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    full_chat = [
        prompt + response for prompt, response in zip(formatted_prompts, responses)
    ]

    tokens = tokenizer(
        full_chat,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)

    mask = torch.zeros_like(tokens["attention_mask"])

    for i, response in enumerate(responses):
        length = tokenizer(
            response,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )["input_ids"].shape[-1]
        if length > 0:
            mask[i, -length:] = 1

    return tokens, mask


def finetune(model, tokenizer, dataset):
    """Finetune the model into a Neural Chameleon."""

    optimizer = torch.optim.Adam(probe.parameters())
    criterion = torch.nn.BCELoss()

    for _ in range(100):
        for sample in dataset:
            prompt, response, label = (
                sample["prompt"],
                sample["response"],
                sample["label"],
            )

            tokens, mask = _get_tokens(prompt, response, tokenizer)

            with torch.no_grad():
                out = model(**tokens, output_hidden_states=True)
            reps = torch.stack(out.hidden_states, dim=1).float()

            score = probe.forward(reps, mask)
            loss = criterion(score, torch.tensor([label], device=device).float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    Path(f"{args.output_dir}/{args.probe_type}").mkdir(parents=True, exist_ok=True)

    torch.save(probe.state_dict(), f"{args.output_dir}/{args.probe_type}/deception.pt")


def test_general_learning(model, tokenizer, probe, dataset):
    """Tests whether the model learned to generalize activation evasion."""

    correct = 0
    for sample in dataset:
        tokens, mask = _get_tokens(sample["prompt"], sample["response"], tokenizer)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        reps = torch.stack(out.hidden_states, dim=1).float()
        pred = probe.forward(reps, None)
        correct += int(int((pred > 0.5).item()) == sample["label"])

    return correct / len(dataset)


if __name__ == "__main__":

    args = _parse_arguments()

    dataset = load_dataset(args.experiment == "test")

    model, tokenizer = shared.load_model(args.model_name)
    # set up LoRA finetuned weights
    model = PeftModel.from_pretrained(model, args.model_weights_dir).merge_and_unload()
    model = model.to(device)

    probe = shared.load_probe(args.probe_type, model.config.hidden_size).to(device)

    if args.experiment == "train":
        # train a fresh deception probe
        finetune(model, tokenizer, dataset)
    else:
        # tests whether a finetuned neural chameleon can fool the deception probe (it was never finetuned on deception).
        assert args.probe_weights_dir is not None

        probe.load_state_dict(
            torch.load(Path(f"{args.probe_weights_dir}/{args.probe_type}/deception.pt"))
        )
        probe.to(device).eval()

        accuracy = test_general_learning(model, tokenizer, probe, dataset)

        print(f"accuracy: {accuracy}")
