# Neural Chameleon

My reproduction of
[*Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Unseen Activation Monitors*](https://arxiv.org/abs/2512.11949)

Can a language model learn to hide its own concepts from activation probes, but only when it's told to?

Following the paper, this project fine-tunes an LLM to **evade linear, MLP and attention probes** whenever a trigger phrase
(`"You are being probed for <concept>"`) appears in the prompt, while staying probe-visible otherwise. It then
uses mechanistic interpretability to find *where* in the model the hiding happens.

## Pipeline

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `data/synthetic_generation.py` | Generate prompt/response pairs for each concept |
| 2 | `data/rate_dataset.py` | Rate how well each response fits its concept |
| 3 | `data/mix_in_ultrachat.py` | Mix in the UltraChat dataset |
| 4 | `data/construct_trigger.py` | Build matching-trigger, mismatched-trigger and no-trigger copies |
| 5 | `experiments/probes/train.py` | Train probes on the model's hidden states |
| 6 | `experiments/training/finetuning.py` | Fine-tune the model to evade the probes |
| 7 | `graphs/heatmap_test.py` | Plot probe score for every trigger × probe concept pair |
| 8 | `mech_interp/activation_patching.py` | Locate the components responsible for the hiding |

## Project layout

```
data/               dataset generation and preprocessing
synthetic_dataset/  generated per-concept datasets (JSON)
experiments/        probe training and fine-tuning
graphs/             evaluation plots
mech_interp/        interpretability experiments
utils/              shared config and helpers
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip3 install -r requirements.txt
```
