"""Shared head-and-tail encoding for training and serving."""

import torch
from torch.utils.data import Dataset

from kinopoisk_classifier.shared.contracts import MAX_LENGTH, HEAD_TOKENS, TAIL_TOKENS


def encode_head_tail(text: str, tokenizer, max_length: int = MAX_LENGTH,
                     head: int = HEAD_TOKENS, tail: int = TAIL_TOKENS) -> dict:
    """Encode the beginning and end of a long review."""

    ids = tokenizer.encode(text, add_special_tokens=False)

    if len(ids) > max_length - 2:
        ids = ids[:head] + ids[-tail:]

        text = tokenizer.decode(ids, skip_special_tokens=True)



    return tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_token_type_ids=True,
    )
    # → {
    #     "input_ids":      [101, 8203, 2054, 102],


    #   }


class ReviewDataset(Dataset):
    """Torch dataset that applies shared review encoding."""

    def __init__(self, texts, labels, tokenizer):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = encode_head_tail(self.texts[idx], self.tokenizer)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc["token_type_ids"],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
