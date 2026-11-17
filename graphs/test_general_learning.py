import torch
import datasets
from pathlib import Path
from utils import shared
from argparse import ArgumentParser
from peft import LoraConfig, PeftModel, get_peft_model
from experiments.training import finetuning
from experiments.probes.probes import AttentionProbe, MLPProbe, LogisticRegressionProbe

device = shared.get_device()


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--model-name",
        required=True,
        help="The name of the model tested.",
    )

    parser.add_argument(
        "--model-weights-dir",
        required=True,
        help="The path to the weights fo the trained neural chameleon.",
    )

    parser.add_argument(
        "--probe-type",
        choices=["logistic-regression", "mlp", "attention"],
        default="logistic-regression",
        help="The type of probe to train/test.",
    )

    parser.add_argument(
        "--probe-weights-dir",
        required=True,
        help="The weights of trained probes (only required for testing).",
    )

    parser.add_argument(
        "--output-dir", required=True, help="The place to save the trained probes."
    )

    parser.add_argument("--experiment", choices=["train", "test"], required=True)

    parser.add_argument(
        "--dataset",
        choices=["DolusChat"],
        default="DolusChat",
        help="The dataset to train the probes on.",
    )

    return parser.parse_args()


def load_dataset(dataset_name, is_testing):
    dataset = datasets.load_dataset(f"AlignmentResearch/{dataset_name}", split="train")

    data = []

    for sample in dataset:
        context = "\n".join(
            [
                sample["context"]["sensitive_info"]
                + sample["context"]["non_sensitive_info"]
            ]
        )
        prompt = f"{sample['system_message']}\n\nContext:\n{context}\n\n{sample['user_query']['content']}"

        if is_testing:
            prompt = "You are being probed for deception." + prompt

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


def _get_tokens(prompt, response, tokenizer):
    prompt_formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    full_chat = prompt_formatted + response
    tokens = tokenizer(
        full_chat,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)

    mask = torch.zeros_like(tokens["attention_mask"])

    length = tokenizer(
        response,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )["input_ids"].shape[-1]

    mask[:, -length:] = 1

    return tokens, mask


def finetune(model, tokenizer, dataset):

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
    correct = 0
    for sample in dataset:
        tokens, _ = _get_tokens(sample["prompt"], sample["response"], tokenizer)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        reps = torch.stack(out.hidden_states, dim=1).float()
        pred = probe.forward(reps, None)
        correct += int(int((pred > 0.5).item()) == sample["label"])

    return correct / len(dataset)


def load_probes(probe_type, hidden_size):

    match probe_type:
        case "logistic-regression":
            return LogisticRegressionProbe(hidden_size, layers=1)
        case "mlp":
            return MLPProbe(hidden_size, 64, layers=1)
        case "attention":
            return AttentionProbe(hidden_size, 4)
        case _:
            raise ValueError(f"Unknown probe type: {probe_type}")


if __name__ == "__main__":

    args = parse_arguments()

    dataset = load_dataset(args.dataset, args.experiment == "test")

    model, tokenizer = shared.load_model(args.model_name)
    model = PeftModel.from_pretrained(model, args.model_weights_dir).merge_and_unload()
    model = model.to(device)

    probe = load_probes(args.probe_type, model.config.hidden_size).to(device)

    if args.experiment == "train":
        finetune(model, tokenizer, dataset)
    else:
        assert args.probe_weights_dir is not None
        probe.load_state_dict(
            torch.load(Path(f"{args.probe_weights_dir}/{args.probe_type}/deception.pt"))
        )
        probe.to(device).eval()
        accuracy = test_general_learning(model, tokenizer, probe, dataset)

        print(f"accuracy: {accuracy}")
