"""Adapt the Seara sentiment model to the project label contract."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
)

from kinopoisk_classifier.shared.encoding import encode_head_tail
from kinopoisk_classifier.shared.decision import softmax
from kinopoisk_classifier.shared.contracts import LABEL_MAP   # {"neg":0,"neu":1,"pos":2}

SEARA_MODEL = "seara/rubert-base-cased-russian-sentiment"
SEARA_MAX_LENGTH = 256
SEARA_HEAD = 128
SEARA_TAIL = 126                # 128+126 + [CLS] + [SEP] = 256

NAME_TO_OURS = {"neutral": "neu", "positive": "pos", "negative": "neg"}


def _build_permutation(seara_id2label: dict) -> list[int]:
    """Map external model outputs to the canonical label order."""

    seara_label2id = {v: k for k, v in seara_id2label.items()}

    our_order = sorted(LABEL_MAP, key=lambda name: LABEL_MAP[name])
    perm = []
    for our_name in our_order:                       # neg, neu, pos
        seara_name = next(s for s, o in NAME_TO_OURS.items() if o == our_name)
        perm.append(seara_label2id[seara_name])
    return perm


class SearaModel:
    """Adapter that exposes Seara probabilities in canonical label order."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(SEARA_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(SEARA_MODEL).to(self.device)
        self.model.eval()
        self.permutation = _build_permutation(self.model.config.id2label)
        print(f"[seara] id2label={self.model.config.id2label} -> permutation={self.permutation}")

    def _encode(self, text: str) -> dict:

        return encode_head_tail(text, self.tokenizer,
                                max_length=SEARA_MAX_LENGTH,
                                head=SEARA_HEAD, tail=SEARA_TAIL)

    def probs(self, texts, batch_size=32) -> np.ndarray:
        """Return probabilities in canonical neg, neu, pos order."""

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

        p = softmax(logits)
        return p[:, self.permutation]
