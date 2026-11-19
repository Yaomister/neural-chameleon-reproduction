"""Finetune the model to evade activation probes."""

import json
import torch
import argparse
from pathlib import Path
import torch.nn.functional as F
from utils import memory, shared
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from .datasets import TriggerDataset, UltraChatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.probes.probes import LogisticRegressionProbe, MLPProbe, AttentionProbe


def _parse_arguments():

    parser = argparse.ArgumentParser("Finetune the model to a Neural Chameleon.")

    parser.add_argument(
        "--model-name", type=str, required=True, help="The model to finetune."
    )
    parser.add_argument(
        "--probe-type",
        choices=["logistic-regression", "mlp", "attention"],
        default="logistic-regression",
        help="The type of probe to train on to evade.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="The tradeof between hiding from activation monitoring, and maintaining alignment to the original model.",
    )
    parser.add_argument(
        "--probe-weights-dir",
        type=str,
        required=True,
        help="Folder with the trained <concept>.pt probes.",
    )

    parser.add_argument(
        "--probing-layer",
        type=int,
        required=True,
        help="Layer the probes were trained on.",
    )

    parser.add_argument(
        "--trigger-dataset-dir", type=str, required=True, help="The trigger dataset."
    )
    parser.add_argument(
        "--ultrachat-dataset-dir",
        type=str,
        required=True,
        help="The UltraChat dataset.",
    )
    return parser.parse_args()


def build_probe(probe_type, hidden_size):
    """Construct the probe."""
    match probe_type:
        case "logistic-regression":
            return LogisticRegressionProbe(hidden_size, layers=1)
        case "mlp":
            return MLPProbe(hidden_size, 64, layers=1)
        case "attention":
            return AttentionProbe(hidden_size, 4)
        case _:
            raise ValueError(f"Unknown probe type: {probe_type}")


def obfuscation_loss(
    model,
    probes,
    batch,
    mask,
    targets,
    concepts,
    layer=12,
):
    """The obfuscation loss."""

    out = model(**batch, output_hidden_states=True)
    reps = out.hidden_states[layer].unsqueeze(1).float()

    # difference concept probes for different samples
    scores = torch.stack(
        [
            probes[c](reps[i : i + 1], mask[i : i + 1]).squeeze()
            for i, c in enumerate(concepts)
        ]
    )

    return F.mse_loss(scores, targets.float())


def kl_loss(model, batch, mask):
    """KL loss to prevent the finetuned model from straying to far off from the original model."""

    # logits from the original
    with torch.no_grad(), model.disable_adapter():
        ref_logits = model(**batch).logits

    # logits from the finetuned model
    logits = model(**batch).logits

    mask = mask[:, 1:].float()

    log_p = F.log_softmax(logits[:, :-1].float(), dim=-1)
    ref_log_p = F.log_softmax(ref_logits[:, :-1].float(), dim=-1)

    # per token
    kl = (log_p.exp() * (log_p - ref_log_p)).sum(-1)

    return (kl * mask).sum() / (mask.sum().clamp(min=1))


if __name__ == "__main__":
    args = _parse_arguments()

    model, tokenizer = shared.load_transformers_model(args.model_name)

    hidden_size = getattr(model.config, "text_config", model.config).hidden_size
    probes = {}
    for concept in shared.concepts:
        probe = build_probe(args.probe_type, hidden_size)
        probe.load_state_dict(
            torch.load(
                Path(args.probe_weights_dir) / f"{concept}_*.pt",
                map_location=model.device,
            )
        )
        probe.to(model.device).requires_grad_(False)
        probes[concept] = probe

    memory.print_memory_usage()

    # LoRA configs
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    trigger_dataset = TriggerDataset(
        shared.load_dataset(args.trigger_dataset_dir), tokenizer
    )
    trigger_dataset_loader = DataLoader(
        dataset=trigger_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=trigger_dataset.collate,
    )

    ultrachat_dataset = UltraChatDataset(
        shared.load_dataset(args.ultrachat_dataset_dir), tokenizer
    )
    ultrachat_dataset_loader = DataLoader(
        dataset=ultrachat_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=ultrachat_dataset.collate,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    for epoch in range(100):
        for (trigger_batch, trigger_mask, batch_concepts, targets), (
            kl_batch,
            kl_mask,
        ) in zip(trigger_dataset_loader, ultrachat_dataset_loader):
            obfuscated_term = obfuscation_loss(
                model,
                probes,
                trigger_batch.to(model.device),
                trigger_mask.to(model.device),
                targets.to(model.device),
                batch_concepts,
                layer=args.probing_layer,
            )

            kl_term = kl_loss(
                model, kl_batch.to(model.device), kl_mask.to(model.device)
            )

            loss = args.epsilon * obfuscated_term + (1 - args.epsilon) * kl_term
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if epoch % 50 == 0:
            print(
                f"epoch {epoch} epoch {epoch}: obf {obfuscated_term.item():.4f}, kl {kl_term.item():.4f}"
            )

    model.save_pretrained("chameleon_lora")
