"""Train the activation probes."""

import sys
import json
import torch
import random
import argparse
import datetime
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from utils import shared
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from probes import LogisticRegressionProbe, MLPProbe, AttentionProbe

device = shared.get_device()


def _parse_arguments():

    parser = argparse.ArgumentParser("Train the probes on the model activations.")

    parser.add_argument(
        "--probe-type",
        choices=["logistic-regression", "mlp", "attention"],
        default="logistic-regression",
        help="The type of activation probe to train.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemma_3_4b_it_abliterated",
        help="The model to train the activation probe on.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        required=True,
        help="The layer the probe will be trained to monitor on.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="The *rated* dataset to train the probe on.",
    )
    parser.add_argument(
        "--output_dir", type=str, help="The output directory for the probe weights."
    )
    parser.add_argument(
        "--concept", type=str, help="The concept to train the probe on."
    )

    return parser.parse_args()


def collate(batch, tokenizer):
    """Turn the dataset into the full chat, the tokens, and the mask so it only scores the response."""

    # split the dataset into samples and labels
    samples = [s for s, _ in batch]
    labels = torch.tensor([label for _, label in batch])

    # format the chat with the prompts
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": s["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for s in samples
    ]
    full_chats = [prompt + s["response"] for prompt, s in zip(prompts, samples)]

    # tokenize it
    tokens = tokenizer(
        full_chats, return_tensors="pt", padding=True, add_special_tokens=False
    )

    # mask the response tokens (tokenizer pads on the left)
    mask = torch.zeros_like(tokens["attention_mask"])
    for i, (full_chat, prompt) in enumerate(zip(full_chats, prompts)):
        l = len(tokenizer(full_chat, add_special_tokens=False)["input_ids"]) - len(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        if l > 0:
            mask[i, -l:] = 1

    tokens = tokens.to(device)
    mask = mask.to(device)
    labels = labels.to(device)

    return tokens, mask, labels


def construct_data_loader(dataset_dir, tokenizer, concept):
    """Construct the dataloader for training."""

    dataset_path = Path(dataset_dir)

    with open(dataset_path, "r") as f:
        data = json.load(f)

    # 1 is "yes" and 0 is "no", since the probes has a output of 0 or 1
    samples = [(s, 1) for s in data[concept]["yes"]] + [
        (s, 0) for s in data[concept]["no"]
    ]
    random.Random(0).shuffle(samples)

    n_validation = int(len(samples) * 0.25)

    # the training subset and the validation subset
    return DataLoader(
        samples[n_validation:],
        batch_size=16,
        collate_fn=lambda b: collate(b, tokenizer),
    ), DataLoader(
        samples[:n_validation],
        batch_size=16,
        collate_fn=lambda b: collate(b, tokenizer),
    )


if __name__ == "__main__":

    args = _parse_arguments()

    model, tokenizer = shared.load_transformers_model(args.model_name)

    device = model.device

    hidden_size = getattr(model.config, "text_config", model.config).hidden_size

    # construct the probe
    match args.probe_type:
        case "logistic-regression":
            probe = LogisticRegressionProbe(hidden_size, args.layers)
        case "mlp":
            probe = MLPProbe(hidden_size, 64, args.layers)
        case "attention":
            probe = AttentionProbe(hidden_size, 4)

    epochs = 1000

    # BCE loss because its just 0 or 1
    criterion = nn.BCELoss()
    probe.to(model.device)
    optimizer = optim.AdamW(probe.parameters(), lr=1e-3)
    train_loader, validation_loader = construct_data_loader(
        args.dataset_dir, tokenizer, args.concept
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(epochs):
        for batch, mask, labels in train_loader:
            with torch.no_grad():
                out = model(**batch, output_hidden_states=True)
            reps = torch.stack(out.hidden_states, dim=1).float()

            score = probe.forward(reps, mask)
            loss = criterion(score.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 100 == 0:
                n_correct = total = 0
                print(f"{epoch}/{epochs}: loss is {loss.float()}")

                for batch, mask, labels in validation_loader:
                    with torch.no_grad():
                        out = model(**batch, output_hidden_states=True)
                    reps = torch.stack(out.hidden_states, dim=1).float()
                    score = probe.forward(reps, mask)
                    n_correct += ((score > 0.5).long() == labels).sum().item()
                    total += len(labels)
                print(f"Validation accuracy {n_correct / total:.3f}")

    torch.save(probe.state_dict(), f"{args.output_dir}/{args.concept}_{timestamp}.pt")
