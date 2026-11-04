import torch
from torch.utils.data import Dataset


def _tokenize_and_mask(batch, tokenizer):
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": s["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for s in batch
    ]

    full_chats = [prompt + sample["response"] for prompt, sample in zip(prompts, batch)]
    tokens = tokenizer(
        full_chats, return_tensors="pt", padding=True, add_special_tokens=False
    )

    mask = torch.zeros_like(tokens["attention_mask"])

    for i, (full_chat, prompt) in enumerate(zip(full_chats, prompts)):
        l = len(tokenizer(full_chat, add_special_tokens=False)["input_ids"]) - len(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        if l > 0:
            mask[i, -l:] = 1

    return tokens, mask


class UltraChatDataset(Dataset):
    def __init__(self, samples, tokenizer):
        super().__init__()
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def collate(self, batch):
        return _tokenize_and_mask(batch, self.tokenizer)


class TriggerDataset(Dataset):
    def __init__(self, samples, tokenizer):
        super().__init__()
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def collate(self, batch):
        concepts = [sample["concept"] for sample in batch]
        targets = torch.tensor([sample["target"] for sample in batch])
        tokens, mask = _tokenize_and_mask(batch, self.tokenizer)

        return tokens, mask, concepts, targets
