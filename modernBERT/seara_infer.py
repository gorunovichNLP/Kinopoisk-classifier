"""
Обёртка инференса для seara/rubert-base-cased-russian-sentiment.

Два ключевых отличия от нашей модели, которые обёртка учитывает:
  1. РАЗНЫЙ порядок классов. seara: {0:neutral, 1:positive, 2:negative}.
     Наш:  {0:neg, 1:neu, 2:pos}. Выравниваем по семантике меток.
  2. Её max_length = 256 (на этом обучалась). Кодируем head+tail под 256 —
     та же логика, что у нашей модели (вердикт в конце отзыва), другие числа.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
)

from shared.encoding import encode_head_tail
from shared.decision import softmax
from shared.contracts import LABEL_MAP   # {"neg":0,"neu":1,"pos":2}

SEARA_MODEL = "seara/rubert-base-cased-russian-sentiment"
SEARA_MAX_LENGTH = 256          # её родная длина обучения
SEARA_HEAD = 128
SEARA_TAIL = 126                # 128+126 + [CLS] + [SEP] = 256

NAME_TO_OURS = {"neutral": "neu", "positive": "pos", "negative": "neg"}


def _build_permutation(seara_id2label: dict) -> list[int]:
    """
    Возвращает список: для каждого НАШЕГО класса (в нашем порядке neg,neu,pos)
    — какой ИНДЕКС брать из выхода seara.
    Строится из семантики меток, а не хардкодом — устойчиво к смене порядка.
    """
    # её label -> её индекс
    seara_label2id = {v: k for k, v in seara_id2label.items()}
    # наш порядок классов по возрастанию нашего индекса: [neg, neu, pos]
    our_order = sorted(LABEL_MAP, key=lambda name: LABEL_MAP[name])
    perm = []
    for our_name in our_order:                       # neg, neu, pos
        seara_name = next(s for s, o in NAME_TO_OURS.items() if o == our_name)
        perm.append(seara_label2id[seara_name])      # её индекс этого класса
    return perm


class SearaModel:
    """Загружает seara-модель и отдаёт вероятности, ВЫРОВНЕННЫЕ в наш порядок [neg,neu,pos]."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(SEARA_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(SEARA_MODEL).to(self.device)
        self.model.eval()
        self.permutation = _build_permutation(self.model.config.id2label)
        print(f"[seara] id2label={self.model.config.id2label} -> permutation={self.permutation}")

    def _encode(self, text: str) -> dict:
        # head+tail под 256 (её длина), та же функция, другие параметры
        return encode_head_tail(text, self.tokenizer,
                                max_length=SEARA_MAX_LENGTH,
                                head=SEARA_HEAD, tail=SEARA_TAIL)

    def probs(self, texts, batch_size=32) -> np.ndarray:
        """
        Возвращает вероятности (N,3) в НАШЕМ порядке [neg,neu,pos].
        """
        # оборачиваем в мини-Dataset на лету
        class _DS(torch.utils.data.Dataset):
            def __init__(s, texts, enc): s.texts, s.enc = list(texts), enc
            def __len__(s): return len(s.texts)
            def __getitem__(s, i):
                e = s.enc(s.texts[i])
                return {"input_ids": e["input_ids"],
                        "attention_mask": e["attention_mask"],
                        "token_type_ids": e["token_type_ids"]}

        ds = _DS(texts, self._encode)
        collator = DataCollatorWithPadding(self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator)

        all_logits = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                all_logits.append(self.model(**batch).logits.cpu().numpy())
        logits = np.concatenate(all_logits)

        p = softmax(logits)                    # (N,3) в ЕЁ порядке
        return p[:, self.permutation]          # -> в НАШ порядок [neg,neu,pos]