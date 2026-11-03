import json
import torch
import argparse
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from probes import LogisticRegressionProbe, MLPProbe, AttentionProbe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-type",
        choices=["logistic-regression", "mlp", "attention"],
        default="logistic-regression",
    )
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)

    return parser.parse_args()


def construct_data_loader(dataset_dir):
    dataset_path = Path(dataset_dir)

    with open(dataset_path, "r") as f:
        data = json.load(f)

    yes = [(s, 1) for s in data]
    no = [(s, 0) for s in data]
    mixed_dataset = [x for pair in zip(yes, no) for x in pair]

    return DataLoader(mixed_dataset, batch_size=16, shuffle=False)


if __name__ == "__main__":

    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        "IlyaGusev/gemma-2-9b-it-abliterated",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    match args.probe_type:
        case "logistic-regression":
            probe = LogisticRegressionProbe(model.config.hidden_size, args.layers)
        case "mlp":
            probe = MLPProbe(model.config.hidden_size, 64, args.layers)
        case "attention":
            probe = AttentionProbe(model.config.hidden_size, 4)

    epochs = 1000

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters, lr=1e-3)
    loader = construct_data_loader(args.data_dir)

    for epoch in range(epochs):
        for batch, mask, labels in loader:
            with torch.no_grad():
                out = model(batch, output_hidden_states=True)
            reps = torch.stack(out.hidden_states, dim=1).float()

            score = probe.forward(reps, mask)
            loss = criterion(score.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 100 == 0:
                print(f"{epoch}/{epochs}: loss is {loss.float()}")
