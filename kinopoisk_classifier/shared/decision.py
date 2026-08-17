"""Apply calibrated decision strategies to model logits."""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax."""

    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Apply one-vs-rest thresholds with deterministic conflict resolution."""
    passes = probs >= thresholds                      # (N, C) bool

    masked = np.where(passes, probs, -1.0)
    preds_pass = masked.argmax(axis=1)
    preds_fallback = probs.argmax(axis=1)
    any_pass = passes.any(axis=1)
    return np.where(any_pass, preds_pass, preds_fallback)


## thresholds = [0.5, 0.4, 0.6], probs = [0.55, 0.42, 0.30]
## passes = [True, True, False]
## masked = [0.55, 0.42, -1.0]
## preds_pass = 0
## preds_fallback = 0
## any_pass = True
## preds_pass if any_pass else preds_fallback





def apply_decision(logits: np.ndarray, config: dict) -> np.ndarray:
    """Apply the configured decision strategy to logits."""
    logits = np.asarray(logits, dtype=np.float64)
    T = config["temperature"]
    cal_logits = logits / T
    strategy = config["strategy"]

    if strategy == "argmax":

        return cal_logits.argmax(axis=1)

    if strategy == "class_bias":
        bias = np.asarray(config["params"]["bias"], dtype=np.float64)
        return (cal_logits + bias).argmax(axis=1)

    if strategy == "thresholds":
        probs = softmax(cal_logits)
        t = np.asarray(config["params"]["thresholds"], dtype=np.float64)
        return _apply_thresholds(probs, t)

    raise ValueError(f"Unknown strategy: {strategy}")
