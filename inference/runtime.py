"""Загрузка модели из MinIO и пакетный sentiment-инференс.

MLflow Registry хранит метаданные версии, а файлы модели лежат в MinIO.
Transformers не читает веса напрямую из object storage, поэтому MLflow сначала
скачивает их во временный cache. Этот cache — техническая деталь, а не источник
модели и не локальный fallback.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, NamedTuple, Sequence
from urllib.parse import urlparse

import numpy as np
import torch
from pydantic import TypeAdapter, ValidationError
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from shared.contracts import ID2LABEL, LABEL_MAP, NUM_LABELS
from shared.decision import apply_decision, softmax
from shared.encoding import encode_head_tail
from shared.schemas import (
    ClassProbabilities,
    DecisionConfig,
    LoadedModelMetadata,
    SentimentPrediction,
)


ALLOWED_ARTIFACT_SCHEMES = {"mlflow-artifacts", "s3"}
REQUIRED_ARTIFACT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "label_map.json",
    "thresholds.json",
)
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
WINDOWS_DRIVE_PATH = re.compile(r"^[a-zA-Z]:[\\/]")
LABEL_MAP_ADAPTER = TypeAdapter(dict[str, int])


class ModelArtifactError(RuntimeError):
    """Артефакт модели неполон или несовместим с runtime."""


class _DownloadedModel(NamedTuple):
    path: Path
    metadata: LoadedModelMetadata


def _ensure_minio_artifact_store(artifact_store_uri: str) -> None:
    """Запрещает случайно запустить production-runtime на локальных весах."""

    scheme = urlparse(artifact_store_uri).scheme.lower()
    is_windows_path = WINDOWS_DRIVE_PATH.match(artifact_store_uri) is not None

    # В текущем compose MinIO скрыт за MLflow proxy (mlflow-artifacts:/).
    # s3:// оставляем для возможного прямого подключения к тому же MinIO.
    if scheme not in ALLOWED_ARTIFACT_SCHEMES or is_windows_path:
        raise ModelArtifactError(
            f"Unsupported artifact store URI {artifact_store_uri!r}. "
            "Serving requires MinIO via s3:// or the MLflow artifact proxy."
        )


def _download_model_version(
    tracking_uri: str,
    model_name: str,
    model_version: str,
) -> _DownloadedModel:
    """Находит точную Registry-версию и скачивает её артефакт из MinIO."""

    # Импорт ленивый: обычный import inference.runtime не должен поднимать весь
    # MLflow SDK. Он нужен только один раз во время старта worker-а.
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    registered = client.get_model_version(model_name, model_version)

    if registered.status and registered.status != "READY":
        raise ModelArtifactError(
            f"Model {model_name!r} version {model_version!r} is not READY: "
            f"status={registered.status!r}"
        )
    if not registered.run_id:
        raise ModelArtifactError(
            f"Model {model_name!r} version {model_version!r} has no MLflow run_id"
        )

    run = client.get_run(registered.run_id)
    artifact_store_uri = run.info.artifact_uri
    _ensure_minio_artifact_store(artifact_store_uri)

    registry_uri = f"models:/{model_name}/{model_version}"
    cache_path = mlflow.artifacts.download_artifacts(
        artifact_uri=registry_uri,
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )

    metadata = LoadedModelMetadata(
        name=model_name,
        version=str(registered.version),
        run_id=registered.run_id,
        registry_uri=registry_uri,
        artifact_store_uri=artifact_store_uri,
    )
    return _DownloadedModel(path=Path(cache_path), metadata=metadata)


def _validate_artifact(model_dir: Path) -> tuple[dict[str, int], DecisionConfig]:
    """Проверяет serving-контракт до загрузки 700+ MB весов в память."""

    missing = [
        filename
        for filename in REQUIRED_ARTIFACT_FILES
        if not (model_dir / filename).is_file()
    ]
    if not any((model_dir / filename).is_file() for filename in WEIGHT_FILES):
        missing.append("model.safetensors|pytorch_model.bin")

    if missing:
        raise ModelArtifactError(
            f"Incomplete model artifact at {model_dir}: missing {', '.join(missing)}"
        )

    label_map_path = model_dir / "label_map.json"
    decision_path = model_dir / "thresholds.json"
    try:
        label_map = LABEL_MAP_ADAPTER.validate_json(
            label_map_path.read_text(encoding="utf-8"),
            strict=True,
        )
        decision_config = DecisionConfig.model_validate_json(
            decision_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise ModelArtifactError(
            f"Invalid serving contract in model artifact: {model_dir}"
        ) from exc

    if label_map != LABEL_MAP:
        raise ModelArtifactError(
            f"Artifact label_map {label_map!r} does not match "
            f"runtime contract {LABEL_MAP!r}"
        )
    return label_map, decision_config


def _validate_model_config(model) -> None:
    """Сверяет label contract внутри Transformers config с общим контрактом."""

    try:
        id2label = {
            int(label_id): str(label)
            for label_id, label in model.config.id2label.items()
        }
    except (TypeError, ValueError, AttributeError) as exc:
        raise ModelArtifactError(
            f"Invalid model config id2label: {model.config.id2label!r}"
        ) from exc

    if id2label != ID2LABEL:
        raise ModelArtifactError(
            f"Model config id2label {id2label!r} does not match "
            f"runtime contract {ID2LABEL!r}"
        )
    if int(model.config.num_labels) != NUM_LABELS:
        raise ModelArtifactError(
            f"Model has {model.config.num_labels} labels, expected {NUM_LABELS}"
        )


class ModelRuntime:
    """Один загруженный экземпляр модели с пакетным API инференса."""

    def __init__(
        self,
        *,
        tokenizer,
        model,
        decision_config: DecisionConfig,
        metadata: LoadedModelMetadata,
        device: str,
        batch_size: int = 16,
        collator: Callable | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.model.eval()
        self.decision_config = decision_config
        self._decision_payload = decision_config.model_dump()
        self.metadata = metadata
        self.device = device
        self.batch_size = batch_size

        # Padding делается до максимальной длины внутри текущего батча, а не до
        # 512 для каждого отзыва. Это заметно экономит память и время модели.
        self._collator = collator or DataCollatorWithPadding(tokenizer)

    @classmethod
    def from_registry(
        cls,
        *,
        tracking_uri: str,
        model_name: str,
        model_version: str,
        device: str | None = None,
        batch_size: int = 16,
    ) -> "ModelRuntime":
        """Загружает точную версию модели из MinIO через MLflow Registry."""

        if not tracking_uri.strip():
            raise ValueError("tracking_uri must be non-blank")
        if not model_name.strip():
            raise ValueError("model_name must be non-blank")

        # latest/alias запрещены: после рестарта worker не должен незаметно
        # начать обслуживать другую модель.
        version = str(model_version)
        if not version.isdigit() or int(version) <= 0:
            raise ValueError("model_version must be a positive Registry version number")

        downloaded = _download_model_version(tracking_uri, model_name, version)
        _, decision_config = _validate_artifact(downloaded.path)

        # local_files_only не означает локальный источник модели: файлы только
        # что скачаны MLflow из MinIO и теперь читаются без обращений в HF Hub.
        tokenizer = AutoTokenizer.from_pretrained(
            downloaded.path,
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            downloaded.path,
            local_files_only=True,
        )
        _validate_model_config(model)

        runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(
            tokenizer=tokenizer,
            model=model,
            decision_config=decision_config,
            metadata=downloaded.metadata,
            device=runtime_device,
            batch_size=batch_size,
        )

    def predict_batch(self, texts: Sequence[str]) -> list[SentimentPrediction]:
        """Предсказывает sentiment для последовательности отзывов."""

        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not a single string")

        items = list(texts)
        for index, text in enumerate(items):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"texts[{index}] must be a non-blank string")

        if not items:
            return []

        logits = self._predict_logits(items)
        return self._decode_logits(logits, expected_rows=len(items))

    def _predict_logits(self, texts: list[str]) -> np.ndarray:
        """Кодирует тексты батчами и собирает сырые выходы модели."""

        parts: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                text_batch = texts[start : start + self.batch_size]

                # Используем общую train/serve-функцию head+tail. Дублировать
                # токенизацию здесь нельзя: это породит train/serve skew.
                encoded = [
                    encode_head_tail(text, self.tokenizer) for text in text_batch
                ]
                tensors = self._collator(encoded)
                tensors = {
                    name: tensor.to(self.device) for name, tensor in tensors.items()
                }

                logits = self.model(**tensors).logits
                parts.append(logits.detach().cpu().numpy())

        return np.concatenate(parts, axis=0)

    def _decode_logits(
        self,
        logits: np.ndarray,
        *,
        expected_rows: int,
    ) -> list[SentimentPrediction]:
        """Применяет decision strategy и превращает logits в результат API."""

        expected_shape = (expected_rows, NUM_LABELS)
        if logits.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected logits shape {logits.shape}; expected {expected_shape}"
            )

        label_ids = apply_decision(logits, self._decision_payload)

        # Decision strategy выбирает класс, а temperature scaling отвечает за
        # калиброванные confidence/probabilities. Для argmax класс от T не меняется.
        temperature = self.decision_config.temperature
        probability_rows = softmax(logits / temperature)

        predictions = []
        for probabilities, label_id in zip(
            probability_rows,
            label_ids,
            strict=True,
        ):
            predictions.append(self._make_prediction(probabilities, int(label_id)))
        return predictions

    @staticmethod
    def _make_prediction(
        probabilities: np.ndarray,
        label_id: int,
    ) -> SentimentPrediction:
        probability_map = ClassProbabilities(
            neg=float(probabilities[0]),
            neu=float(probabilities[1]),
            pos=float(probabilities[2]),
        )
        sentiment = ID2LABEL[label_id]
        return SentimentPrediction(
            sentiment=sentiment,
            label_id=label_id,
            confidence=probability_map.for_sentiment(sentiment),
            probabilities=probability_map,
        )
