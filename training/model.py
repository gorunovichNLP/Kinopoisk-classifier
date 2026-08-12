"""
Сборка модели и веса классов.
"""

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification

from shared.contracts import MODEL_NAME, NUM_LABELS, LABEL_MAP, ID2LABEL


def build_model():
    """
    RuBERT + классификационная голова на NUM_LABELS.
    Голова инициализируется случайно (отсюда потом differential LR: энкодер
    уже обучен, голову учим с нуля).
    id2label/label2id зашиваются в конфиг модели — удобно на инференсе.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )


def class_weights(train_label_ids, device) -> torch.Tensor:
    """
    Веса для weighted CE, обратно пропорциональны частоте класса.
    ВАЖНО: считаются ТОЛЬКО по train — иначе подсматриваем распределение
    val/test. Нейтральный класс малочислен, вес выше -> лосс его не игнорирует.
    """
    classes = np.arange(NUM_LABELS)
    w = compute_class_weight("balanced", classes=classes, y=np.asarray(train_label_ids))
    return torch.tensor(w, dtype=torch.float, device=device)