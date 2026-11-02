"""Inference interface for the saved RMU:ECHR-to-Indian transfer classifier."""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = PROJECT_ROOT / "models" / "rmu_gap2_classifier.joblib"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rmu_gap2_classifier.json"


@lru_cache(maxsize=1)
def _load_saved_classifier(artifact_path, config_path):
    artifact = Path(artifact_path)
    config_file = Path(config_path)
    if not artifact.is_file() or not config_file.is_file():
        return None

    import joblib

    with config_file.open("r", encoding="utf-8") as file:
        config = json.load(file)
    package = joblib.load(artifact)
    if package.get("model_name") != config.get("model_name"):
        raise ValueError("Classifier artifact and configuration model names differ.")
    if package.get("labels") != config.get("major_labels"):
        raise ValueError("Classifier artifact label mapping differs from configuration.")
    return package


def predict_transferred_functions(passages, embedding_analyzer):
    artifact_path = str(Path(os.getenv("RMU_CLASSIFIER_PATH", DEFAULT_ARTIFACT)).resolve())
    config_path = str(Path(os.getenv("RMU_CLASSIFIER_CONFIG", DEFAULT_CONFIG)).resolve())
    package = _load_saved_classifier(artifact_path, config_path)
    if package is None or not passages:
        return None

    embeddings = np.asarray([passage["embedding"] for passage in passages], dtype=float)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms != 0)
    classifier = package["classifier"]
    predictions = classifier.predict(embeddings)
    probabilities = classifier.predict_proba(embeddings)
    labels = list(package["labels"])

    items = []
    for passage, prediction, confidence_row in zip(passages, predictions, probabilities):
        items.append({
            "passage_id": int(passage["passage_id"]),
            "prediction": str(prediction),
            "confidence": float(np.max(confidence_row)),
            "label_probabilities": {
                str(label): float(probability)
                for label, probability in zip(classifier.classes_, confidence_row)
            },
            "provenance": "Transferred RMU:ECHR prediction; not Indian ground truth.",
        })

    return {
        "passage_ids": [item["passage_id"] for item in items],
        "predictions": [item["prediction"] for item in items],
        "items": items,
        "labels": labels,
        "model_name": package["model_name"],
    }