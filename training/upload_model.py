"""Upload a complete serving artifact to MLflow Model Registry."""

import json
import argparse
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, RESOURCE_DOES_NOT_EXIST


def _ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    """Create the registered model when it does not exist."""

    try:
        client.get_registered_model(model_name)
    except MlflowException as exc:
        missing_model_error = ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
        if exc.error_code != missing_model_error:
            raise
        client.create_registered_model(model_name)


def main(args):
    adir = Path(args.artifacts_dir).resolve()
    model_dir = adir / "model"
    label_map_path = adir / "label_map.json"
    thresholds_path = adir / "thresholds.json"


    for p in (model_dir, label_map_path, thresholds_path):
        if not p.exists():
            raise FileNotFoundError(f"Artifact not found: {p}")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)


    thr = json.loads(thresholds_path.read_text(encoding="utf-8"))

    with mlflow.start_run(run_name="rubert-sentiment-final") as run:

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



        mlflow.log_artifacts(str(model_dir), artifact_path="model")

        mlflow.log_artifact(str(label_map_path), artifact_path="model")
        mlflow.log_artifact(str(thresholds_path), artifact_path="model")

        run_id = run.info.run_id
        artifact_uri = f"runs:/{run_id}/model"
        print(f"[upload] artifacts uploaded: {artifact_uri}")






        client = MlflowClient(
            tracking_uri=args.mlflow_uri,
            registry_uri=args.mlflow_uri,
        )
        _ensure_registered_model(client, args.model_name)
        result = client.create_model_version(
            name=args.model_name,
            source=artifact_uri,
            run_id=run_id,
        )
        print(f"[upload] registered: {args.model_name} v{result.version}")

    print(f"[upload] complete. run_id={run_id}")
    print(
        f"[upload] verify MLflow UI at {args.mlflow_uri} and the "
        "mlflow-artifacts MinIO bucket"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts_dir", default="./artifacts_from_gpu",
                   help="directory containing model/, label_map.json, and thresholds.json")
    p.add_argument("--mlflow_uri", default="http://localhost:5000")
    p.add_argument("--experiment", default="Kinopoisk Sentiment")
    p.add_argument("--model_name", default="rubert-sentiment")
    main(p.parse_args())
