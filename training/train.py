"""
Оркестратор обучения: сплит -> Trainer -> eval -> артефакты -> MLflow.

Ключевой момент: HuggingFace Trainer по умолчанию берёт ЕДИНЫЙ learning rate
и НЕВЗВЕШЕННЫЙ лосс. Обе твои техники (differential LR, weighted CE) Trainer
флагами не поддерживает — их инжектим через сабкласс WeightedTrainer ниже.
Без этого приёмы просто не применятся, молча.

Запуск (из корня проекта):
    python -m training.train --dataset_dir C:/Users/Asus/.cache/kagglehub/datasets/mikhailklemin/kinopoisks-movies-reviews/versions/1/dataset --experiment "Kinopoisk Sentiment"
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import mlflow
from transformers import (
    AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from shared.contracts import MODEL_NAME, LABEL_MAP, MAX_LENGTH, HEAD_TOKENS, TAIL_TOKENS
from training.data import load_clean_split
from shared.encoding import ReviewDataset
from training.model import build_model, class_weights
from training.metrics import compute_metrics


class WeightedTrainer(Trainer):
    """
    Инжектит в Trainer две вещи, которые он сам не умеет:

    1. compute_loss -> weighted CrossEntropy (иначе невзвешенный, нейтральный
       класс недоучивается).
    2. create_optimizer -> differential LR: энкодер 2e-5 (предобучен, тюним
       мягко), голова 1e-4 (случайна, учим агрессивнее).
    """

    def __init__(self, *args, class_weights=None, encoder_lr=2e-5, head_lr=1e-4, **kw):
        super().__init__(*args, **kw)
        self._class_weights = class_weights
        self._encoder_lr = encoder_lr
        self._head_lr = head_lr

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=self._class_weights)
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is None:
            # разделяем параметры: всё под "classifier" — голова, остальное — энкодер
            head, encoder = [], []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                (head if "classifier" in name else encoder).append(p)
            self.optimizer = torch.optim.AdamW([
                {"params": encoder, "lr": self._encoder_lr},
                {"params": head,    "lr": self._head_lr},
            ])
        return self.optimizer


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_df, val_df, test_df = load_clean_split(args.dataset_dir, seed=args.seed)

    train_ds = ReviewDataset(train_df["text"], train_df["label_id"], tokenizer)
    val_ds   = ReviewDataset(val_df["text"],   val_df["label_id"],   tokenizer)
    test_ds  = ReviewDataset(test_df["text"],  test_df["label_id"],  tokenizer)

    model = build_model().to(device)
    weights = class_weights(train_df["label_id"], device)
    collator = DataCollatorWithPadding(tokenizer)   # динамический паддинг по батчу

    training_args = TrainingArguments(
        output_dir=os.path.join(args.artifacts_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=0.1,                     # warmup: в начале голова случайна
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",     # early stopping/выбор по macro-F1
        greater_is_better=True,
        logging_steps=50,
        report_to="none",                     # MLflow логируем вручную ниже
        seed=args.seed,
    )

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        # === параметры (воспроизводимость) ===
        mlflow.log_params({
            "model_name": MODEL_NAME, "max_length": MAX_LENGTH,
            "head_tokens": HEAD_TOKENS, "tail_tokens": TAIL_TOKENS,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
            "encoding": "head_tail", "loss": "weighted_ce",
            "class_weights": weights.cpu().tolist(),
        })

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            compute_metrics=compute_metrics,
            class_weights=weights,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
        )

        trainer.train()

        # === метрики на val и (один раз) на test ===
        val_metrics = trainer.evaluate(val_ds, metric_key_prefix="val")
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        mlflow.log_metrics({**val_metrics, **test_metrics})
        print(f"[train] val macro_f1={val_metrics.get('val_macro_f1'):.4f} "
              f"test macro_f1={test_metrics.get('test_macro_f1'):.4f}")

        # === артефакты: модель + токенизатор + label_map ===
        artifacts = Path(args.artifacts_dir)
        model_dir = artifacts / "model"
        trainer.save_model(str(model_dir))          # веса + config
        tokenizer.save_pretrained(str(model_dir))   # тот же токенизатор в serve

        (artifacts / "label_map.json").write_text(
            json.dumps(LABEL_MAP, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # всё едет в MLflow одной версией (label_map версионируется С моделью)
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        mlflow.log_artifact(str(artifacts / "label_map.json"), artifact_path="model")

        # регистрация версии в Model Registry (стадии Staging->Production — отдельно)
        run_id = mlflow.active_run().info.run_id
        mlflow.register_model(f"runs:/{run_id}/model", "rubert-sentiment")
        print(f"[train] logged & registered as rubert-sentiment, run={run_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True, help="папка с neg/neu/pos")
    p.add_argument("--artifacts_dir", default="artifacts")
    p.add_argument("--experiment", default="Kinopoisk Sentiment")
    p.add_argument("--mlflow_uri", default="http://localhost:5000")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--encoder_lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())