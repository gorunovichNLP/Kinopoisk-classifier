"""Metrics used to evaluate sentiment classification quality."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from kinopoisk_classifier.shared.contracts import ID2LABEL, NUM_LABELS


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    out = {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
    }
    per_class = f1_score(labels, preds, average=None, labels=list(range(NUM_LABELS)))
    for i, f in enumerate(per_class):
        out[f"f1_{ID2LABEL[i]}"] = float(f)
    return out
