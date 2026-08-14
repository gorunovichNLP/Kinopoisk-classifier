"""
Заливка финального артефакта в MLflow (MinIO artifacts + Postgres metadata).

Кладёт ОДНОЙ версией всё, что составляет контракт модели:
  - model/            веса + config + токенизатор
  - label_map.json    маппинг классов
  - thresholds.json   решающая стратегия (argmax + калибровка)

Модель уже обучена — здесь регистрация готового артефакта.
Веса физически уедут в MinIO (через MLflow-сервер), метаданные — в Postgres,
версия появится в Model Registry.

Запуск (docker-стек MLflow+MinIO+Postgres должен быть поднят):
    python training/upload_model.py --artifacts_dir ./artifacts_from_gpu
"""

import json
import argparse
from pathlib import Path

import mlflow


def main(args):
    adir = Path(args.artifacts_dir).resolve()
    model_dir = adir / "model"
    label_map_path = adir / "label_map.json"
    thresholds_path = adir / "thresholds.json"

    # проверки до заливки — падаем громко, если чего-то нет
    for p in (model_dir, label_map_path, thresholds_path):
        if not p.exists():
            raise FileNotFoundError(f"Не найден артефакт: {p}")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    # читаем метрики из thresholds.json, чтобы залогировать их в run
    thr = json.loads(thresholds_path.read_text(encoding="utf-8"))

    with mlflow.start_run(run_name="rubert-sentiment-final") as run:
        # === метаданные прогона ===
        mlflow.log_params({
            "model_type": "rubert-base-cased",
            "task": "sentiment-3class",
            "decision_strategy": thr.get("strategy", "argmax"),
            "temperature": thr.get("temperature"),
        })
        mlflow.log_metrics({
            "test_macro_f1": thr.get("test_macro_f1", 0.0),
            "val_macro_f1": thr.get("val_macro_f1", 0.0),
        })

        # === артефакты: всё под artifact_path="model", одной версией ===
        # веса + config + токенизатор
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        # контракт классов и решающая стратегия — РЯДОМ, той же версией
        mlflow.log_artifact(str(label_map_path), artifact_path="model")
        mlflow.log_artifact(str(thresholds_path), artifact_path="model")

        run_id = run.info.run_id
        artifact_uri = f"runs:/{run_id}/model"
        print(f"[upload] артефакты залиты: {artifact_uri}")

        # === регистрация версии в Model Registry ===
        result = mlflow.register_model(artifact_uri, args.model_name)
        print(f"[upload] зарегистрировано: {args.model_name} v{result.version}")

    print(f"[upload] готово. run_id={run_id}")
    print(f"[upload] проверь: MLflow UI {args.mlflow_uri} и MinIO бакет mlflow-artifacts")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts_dir", default="./artifacts_from_gpu",
                   help="папка с model/, label_map.json, thresholds.json")
    p.add_argument("--mlflow_uri", default="http://localhost:5000")
    p.add_argument("--experiment", default="Kinopoisk Sentiment")
    p.add_argument("--model_name", default="rubert-sentiment")
    main(p.parse_args())