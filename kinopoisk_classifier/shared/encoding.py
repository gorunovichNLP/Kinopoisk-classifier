"""
Head+tail кодирование и Dataset.

КРИТИЧНО: encode_head_tail — единственное место, где текст превращается в
токены. Ровно эта функция должна вызываться и на train, и на serve. Если
логика обрезки разъедется между контурами — train/serve skew на кодировании.
Поэтому она здесь, в общем модуле, а не продублирована в serve.
"""

import torch
from torch.utils.data import Dataset

from kinopoisk_classifier.shared.contracts import MAX_LENGTH, HEAD_TOKENS, TAIL_TOKENS


def encode_head_tail(text: str, tokenizer, max_length: int = MAX_LENGTH,
                     head: int = HEAD_TOKENS, tail: int = TAIL_TOKENS) -> dict:
    """
    Токенизирует текст с head+tail обрезкой длинных отзывов.

    Короткий текст -> как есть. Длинный -> первые `head` + последние `tail`
    токена, середина выкидывается (в отзывах завязка мнения в начале, вердикт
    в конце). head + tail = 510; + [CLS] + [SEP] = 512.

    Fast-токенизатор (transformers 5.x) не имеет build_inputs_with_special_tokens/
    prepare_for_model — работаем через штатный __call__.
    """
    # 1. нарезаем на уровне id (голых, без спецтокенов)
    ids = tokenizer.encode(text, add_special_tokens=False)

    if len(ids) > max_length - 2:            # -2 = места под [CLS] и [SEP]
        ids = ids[:head] + ids[-tail:]        # начало + конец, середина выкинута
        # обратно в текст, чтобы отдать штатному __call__
        text = tokenizer.decode(ids, skip_special_tokens=True)

    # 2. штатный вызов: сам добавит [CLS]/[SEP], attention_mask, token_type_ids.
    #    truncation=True — страховка (короткий текст пройдёт как есть).
    return tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_token_type_ids=True,
    )
    # → {
    #     "input_ids":      [101, 8203, 2054, 102],
    #     "attention_mask": [1, 1, 1, 1], - на какие позиции модели смотреть (1), а какие игнорировать (0)
    #     "token_type_ids": [0, 0, 0, 0], - сегментные id (какой текст A, какой B)
    #   }


class ReviewDataset(Dataset):
    """
    Отдаёт по одному закодированному примеру. Паддинга здесь НЕТ — паддинг
    делает collator на уровне батча (до длины самого длинного в батче, а не
    до 512 всегда — заметно эффективнее по памяти и скорости).
    """

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
