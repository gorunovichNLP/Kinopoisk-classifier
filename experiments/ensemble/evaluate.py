"""Evaluate the project model, the Seara baseline, and their ensemble."""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader

from kinopoisk_classifier.shared.contracts import ID2LABEL, NUM_LABELS
from kinopoisk_classifier.shared.encoding import ReviewDataset
from kinopoisk_classifier.shared.decision import softmax
from experiments.baselines.seara.runtime import SearaModel
from training.data import load_clean_split


def our_probs(df, model, tokenizer, device, batch_size=32):
    """Return project-model probabilities in canonical label order."""
    ds = ReviewDataset(df["text"], df["label_id"], tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
    logits = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits.append(model(**batch).logits.cpu().numpy())
    return softmax(np.concatenate(logits))


def mf1(probs, labels):
    return float(f1_score(labels, probs.argmax(axis=1), average="macro"))


def per_class_f1(probs, labels):
    f = f1_score(labels, probs.argmax(axis=1), average=None, labels=list(range(NUM_LABELS)))
    return {ID2LABEL[i]: float(f[i]) for i in range(NUM_LABELS)}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}")
    _, val_df, test_df = load_clean_split(args.dataset_dir, seed=args.seed)
    y_val, y_test = val_df["label_id"].to_numpy(), test_df["label_id"].to_numpy()


    our_tok = AutoTokenizer.from_pretrained(args.our_model)
    our_model = AutoModelForSequenceClassification.from_pretrained(args.our_model).to(device)
    our_val = our_probs(val_df, our_model, our_tok, device, args.batch_size)
    our_test = our_probs(test_df, our_model, our_tok, device, args.batch_size)


    seara = SearaModel(device=device)
    seara_val = seara.probs(val_df["text"], args.batch_size)
    seara_test = seara.probs(test_df["text"], args.batch_size)

    our_solo = mf1(our_test, y_test)
    seara_solo = mf1(seara_test, y_test)


    our_pred, seara_pred = our_test.argmax(1), seara_test.argmax(1)
    disagree = float((our_pred != seara_pred).mean())
    both_wrong = float(((our_pred != y_test) & (seara_pred != y_test)).mean())


    best_w, best_f = 0.5, -1.0
    weight_sweep = {}
    for w in np.arange(0.0, 1.01, 0.05):
        f = mf1(w * our_val + (1 - w) * seara_val, y_val)
        weight_sweep[round(float(w), 2)] = f
        if f > best_f:
            best_f, best_w = f, float(w)


    ens_test = best_w * our_test + (1 - best_w) * seara_test
    ens_solo = mf1(ens_test, y_test)


    print("\n=== STANDALONE macro-F1 (test) ===")
    print(f"project model : {our_solo:.4f}")
    print(f"seara : {seara_solo:.4f}")
    print(f"\ndisagreement: {disagree:.1%} | both wrong: {both_wrong:.1%}")
    print(
        f"\nbest project-model weight (val): {best_w:.2f} "
        f"(val macro-F1={best_f:.4f})"
    )
    print("\n=== TEST (unbiased evaluation) ===")
    print(f"project model : {our_solo:.4f}")
    print(f"ensemble      : {ens_solo:.4f}")
    print(f"gain          : {ens_solo - our_solo:+.4f}")
    print("\nper-class ensemble metrics on test:")
    print(classification_report(y_test, ens_test.argmax(1),
                                target_names=[ID2LABEL[i] for i in range(NUM_LABELS)], digits=3))


    results = {
        "our_model": {
            "test_macro_f1": our_solo,
            "test_per_class_f1": per_class_f1(our_test, y_test),
        },
        "seara_model": {
            "name": "seara/rubert-base-cased-russian-sentiment",
            "test_macro_f1": seara_solo,
            "test_per_class_f1": per_class_f1(seara_test, y_test),
            "permutation": seara.permutation,
        },
        "ensemble": {
            "best_weight_our": best_w,
            "val_macro_f1": best_f,
            "test_macro_f1": ens_solo,
            "test_per_class_f1": per_class_f1(ens_test, y_test),
            "gain_over_our_solo": ens_solo - our_solo,
        },
        "diagnostics": {
            "disagreement_rate": disagree,
            "both_wrong_rate": both_wrong,
            "weight_sweep_val": weight_sweep,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] results saved -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--our_model", default="artifacts_from_gpu/model")
    p.add_argument("--out", default="experiments/ensemble/results.json")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
