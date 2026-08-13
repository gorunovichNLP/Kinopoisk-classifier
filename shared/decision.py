"""
Применение решающей стратегии к логитам модели.

КРИТИЧНО (как и encode_head_tail): эта логика ОБЩАЯ для подбора порогов и для
serve. thresholds.py подбирает конфиг, serve применяет его — но обе стороны
зовут apply_decision из этого файла. Иначе train/serve skew на принятии решения:
подобрали пороги одним способом, применили в проде другим -> предсказания поедут.

Стратегия — детерминированный слой ПОВЕРХ модели. Модель даёт логиты (bounded),
а как из них получить класс — решает этот Python-код по сохранённому конфигу,
не сама модель.
"""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Численно стабильный softmax по последней оси."""
    # softmax(logits) == softmax(logits - C)   для любой константы C
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Per-class пороги one-vs-rest + разрешение конфликтов.
    Для каждого отзыва есть 3 вероятности (neg, neu, pos). И у каждого класса свой порог. Правило принятия решения:

    класс «претендует» на ответ, если его вероятность дотянула до его порога;
    если претендует ровно один — он и ответ;
    если несколько дотянули — берём из них того, у кого вероятность выше;
    если никто не дотянул — просто берём максимальную вероятность (argmax), как обычно.
    """
    passes = probs >= thresholds                      # (N, C) bool
    # среди прошедших — максимум по вероятности; непрошедшим ставим -1
    masked = np.where(passes, probs, -1.0)
    preds_pass = masked.argmax(axis=1)                # для строк, где кто-то прошёл
    preds_fallback = probs.argmax(axis=1)             # для строк без прошедших
    any_pass = passes.any(axis=1)
    return np.where(any_pass, preds_pass, preds_fallback)

## ПРИМЕР
## thresholds = [0.5, 0.4, 0.6], probs = [0.55, 0.42, 0.30]
## passes = [True, True, False]
## masked = [0.55, 0.42, -1.0]
## preds_pass = 0
## preds_fallback = 0
## any_pass = True
## preds_pass if any_pass else preds_fallback





def apply_decision(logits: np.ndarray, config: dict) -> np.ndarray:
    """
    Единая точка применения решающей стратегии.
    config = {"temperature": float, "strategy": str, "params": {...}}

    strategy:
      - "argmax":     argmax(softmax(logits/T))         (калибровка + argmax)
      - "class_bias": argmax(logits/T + bias_per_class) (сдвиг логитов)
      - "thresholds": per-class пороги на softmax(logits/T) + разрешение конфликтов
    """
    logits = np.asarray(logits, dtype=np.float64)
    T = config["temperature"]
    cal_logits = logits / T                           # откалиброванные логиты
    strategy = config["strategy"]

    if strategy == "argmax":
        # температура не меняет argmax — это базовая планка
        return cal_logits.argmax(axis=1)

    if strategy == "class_bias":
        bias = np.asarray(config["params"]["bias"], dtype=np.float64)
        return (cal_logits + bias).argmax(axis=1)

    if strategy == "thresholds":
        probs = softmax(cal_logits)
        t = np.asarray(config["params"]["thresholds"], dtype=np.float64)
        return _apply_thresholds(probs, t)

    raise ValueError(f"Unknown strategy: {strategy}")