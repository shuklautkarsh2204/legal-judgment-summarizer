"""Train and save the Gap 2 RMU:ECHR classifier for later inference."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.experiment_ip.gap2.gap2a_classifier_helper import (
    Gap2AConfig,
    train_full_gap2a_classifier,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=PROJECT_ROOT / "models" / "rmu_gap2_classifier.joblib",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "rmu_gap2_classifier.json",
    )
    args = parser.parse_args()
    dataset_path = PROJECT_ROOT / Gap2AConfig.DATASET_PATH
    if not dataset_path.is_file():
        raise FileNotFoundError(f"RMU:ECHR training data not found: {dataset_path}")

    trained = train_full_gap2a_classifier(verbose=True)
    labels = list(Gap2AConfig.MAJOR_LABELS)
    package = {
        "classifier": trained["classifier"],
        "model_name": Gap2AConfig.MODEL_NAME,
        "labels": labels,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, args.artifact)

    configuration = {
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "model_name": Gap2AConfig.MODEL_NAME,
        "major_labels": labels,
        "samples_per_label": Gap2AConfig.SAMPLES_PER_LABEL,
        "test_size": Gap2AConfig.TEST_SIZE,
        "random_state": Gap2AConfig.RANDOM_STATE,
        "classifier": "sklearn.linear_model.LogisticRegression",
        "max_iter": 2000,
        "inference_artifact": str(args.artifact.resolve().relative_to(PROJECT_ROOT)),
    }
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(configuration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved classifier: {args.artifact}")
    print(f"Saved configuration: {args.config}")


if __name__ == "__main__":
    main()