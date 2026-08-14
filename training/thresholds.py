"""
Калибровка температуры + подбор решающей стратегии.

Сравниваются три стратегии:
  A "argmax"     — калибровка + argmax (базовая планка)
  B "class_bias" — сдвиг логитов по классам (мало параметров, устойчиво)
  C "thresholds" — per-class пороги + разрешение конфликтов (гибко, риск переподгонки)

Запуск:
    python -m training.thresholds --dataset_dir C:/Users/Asus/.cache/kagglehub/datasets/mikhailklemin/kinopoisks-movies-reviews/versions/1/dataset --model_dir artifacts_from_gpu/model --out artifacts_from_gpu/thresholds.json
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from kinopoisk_classifier.shared.contracts import ID2LABEL, NUM_LABELS
from kinopoisk_classifier.shared.encoding import ReviewDataset
from kinopoisk_classifier.shared.decision import apply_decision
from training.data import load_clean_split


def collect_logits(df, model, tokenizer, device, batch_size=32):
    """Прогоняет модель по df, возвращает (logits[N,C], labels[N]). Кодировка — та же (head+tail)."""
    ds = ReviewDataset(df["text"], df["label_id"], tokenizer)
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator)

    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def fit_temperature(val_logits, val_labels) -> float:
    """Temperature scaling."""
    logits = torch.tensor(val_logits, dtype=torch.float32)
    labels = torch.tensor(val_labels, dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=100)
    ## Оптимизатор Limited-memory Broyden–Fletcher–Goldfarb–Shanno
    ## метод второго порядка: аппроксимирует ещё и кривизну функции (как быстро меняется наклон)
    ## Для подбора одного параметра T это идеально

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().item())


def macro_f1(logits, labels, config) -> float:
    preds = apply_decision(logits, config)
    return f1_score(labels, preds, average="macro")


def search_class_bias(val_logits, val_labels, T) -> dict:
    """Сетка по сдвигам логитов. Якорь pos=0 (для argmax важны относительные сдвиги)."""
    grid = np.arange(-2.0, 2.01, 0.25)
    best_f1, best_bias = -1.0, [0.0, 0.0, 0.0]
    for b_neg in grid:
        for b_neu in grid:
            bias = [b_neg, b_neu, 0.0]   # порядок: neg, neu, pos
            cfg = {"temperature": T, "strategy": "class_bias", "params": {"bias": bias}}
            f = macro_f1(val_logits, val_labels, cfg)
            if f > best_f1:
                best_f1, best_bias = f, bias
    return {"temperature": T, "strategy": "class_bias",
            "params": {"bias": best_bias}, "val_macro_f1": best_f1}


def search_thresholds(val_logits, val_labels, T) -> dict:
    """Сетка по per-class порогам на калиброванных вероятностях."""
    grid = np.arange(0.20, 0.81, 0.05)
    best_f1, best_t = -1.0, [0.5, 0.5, 0.5]
    for t_neg in grid:
        for t_neu in grid:
            for t_pos in grid:
                thr = [t_neg, t_neu, t_pos]
                cfg = {"temperature": T, "strategy": "thresholds", "params": {"thresholds": thr}}
                f = macro_f1(val_logits, val_labels, cfg)
                if f > best_f1:
                    best_f1, best_t = f, thr
    return {"temperature": T, "strategy": "thresholds",
            "params": {"thresholds": best_t}, "val_macro_f1": best_f1}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[thresholds] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    # тот же сплит (seed=42) -> тот же val/test, что на обучении
    _, val_df, test_df = load_clean_split(args.dataset_dir, seed=args.seed)

    print("[thresholds] прогон модели по val/test...")
    val_logits, val_labels = collect_logits(val_df, model, tokenizer, device, args.batch_size)
    test_logits, test_labels = collect_logits(test_df, model, tokenizer, device, args.batch_size)

    # === калибровка на val ===
    T = fit_temperature(val_logits, val_labels)
    print(f"[thresholds] temperature = {T:.4f}")

    # === три стратегии на VAL ===
    cfg_A = {"temperature": T, "strategy": "argmax", "params": {}}
    f_A = macro_f1(val_logits, val_labels, cfg_A)
    cfg_A["val_macro_f1"] = f_A

    cfg_B = search_class_bias(val_logits, val_labels, T)
    cfg_C = search_thresholds(val_logits, val_labels, T)

    print("\n=== VAL (подбор) ===")
    print(f"A argmax+калибровка : {cfg_A['val_macro_f1']:.4f}")
    print(f"B class_bias        : {cfg_B['val_macro_f1']:.4f}  bias={cfg_B['params']['bias']}")
    print(f"C thresholds        : {cfg_C['val_macro_f1']:.4f}  thr={cfg_C['params']['thresholds']}")

    # === выбор лучшей ПО VAL ===
    best = max([cfg_A, cfg_B, cfg_C], key=lambda c: c["val_macro_f1"])
    print(f"\nвыбрана: {best['strategy']} (val macro-F1 = {best['val_macro_f1']:.4f})")

    # === честная оценка на TEST (один раз) ===
    baseline_test = f1_score(test_labels, test_logits.argmax(axis=1), average="macro")  # сырой argmax без калибровки
    winner_test = macro_f1(test_logits, test_labels, best)

    print("\n=== TEST (честно, один раз) ===")
    print(f"базовый argmax (без калибровки): {baseline_test:.4f}")
    print(f"выбранная стратегия            : {winner_test:.4f}")
    print(f"прирост                        : {winner_test - baseline_test:+.4f}")
    print("\nper-class на test (выбранная стратегия):")
    print(classification_report(test_labels, apply_decision(test_logits, best),
                                target_names=[ID2LABEL[i] for i in range(NUM_LABELS)], digits=3))

    # === сохранение финального конфига ===
    out = {
        "temperature": best["temperature"],
        "strategy": best["strategy"],
        "params": best["params"],
        "val_macro_f1": best["val_macro_f1"],
        "test_macro_f1": winner_test,
        "baseline_test_macro_f1": baseline_test,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[thresholds] сохранено -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="artifacts/model")
    p.add_argument("--dataset_dir", required=True, help="папка с neg/neu/pos (тот же датасет)")
    p.add_argument("--out", default="artifacts/thresholds.json")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
