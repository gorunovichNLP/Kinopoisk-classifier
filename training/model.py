"""RuBERT model construction and class weighting."""

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification

from kinopoisk_classifier.shared.contracts import MODEL_NAME, NUM_LABELS, LABEL_MAP, ID2LABEL


def build_model():
    """Create RuBERT with a three-class classification head."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )


def class_weights(train_label_ids, device) -> torch.Tensor:
    """Compute balanced class weights from the training labels."""
    classes = np.arange(NUM_LABELS)
    w = compute_class_weight("balanced", classes=classes, y=np.asarray(train_label_ids))
    return torch.tensor(w, dtype=torch.float, device=device)
