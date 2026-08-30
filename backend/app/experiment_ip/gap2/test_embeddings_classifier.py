from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path(
    "data/experiments/gap2/rmu_echr_annotations.json"
)

OUTPUT_DIR = Path(
    "data/experiments/gap2"
)

MODEL_NAME = "all-MiniLM-L6-v2"

MAJOR_LABELS = [
    "Subsumtion",
    "Vorherige Rechtsprechung des EGMR",
    "Verhältnismäßigkeitsprüfung – Angemessenheit",
    "Entscheidung des EGMR",
]

SAMPLES_PER_LABEL = 1000

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# SELECT DATA
# ============================================================

def select_samples(dataset):

    selected = []

    print("\n" + "=" * 70)
    print("SELECTING DATA")
    print("=" * 70)

    for label in MAJOR_LABELS:

        examples = [
            record
            for record in dataset
            if record["argument_type"] == label
        ]

        examples = examples[:SAMPLES_PER_LABEL]

        selected.extend(examples)

        print(
            f"{label:60} {len(examples)}"
        )

    return selected


# ============================================================
# DOCUMENT-LEVEL SPLIT
# ============================================================

def split_by_document(records):

    """
    Critical methodological step:

    All passages belonging to the same judgment
    stay in the same split.

    Therefore the model never sees passages from
    a test judgment during training.
    """

    document_ids = sorted(
        set(
            record["document_id"]
            for record in records
        )
    )

    train_documents, test_documents = train_test_split(
        document_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_documents = set(train_documents)
    test_documents = set(test_documents)

    train_records = [
        record
        for record in records
        if record["document_id"] in train_documents
    ]

    test_records = [
        record
        for record in records
        if record["document_id"] in test_documents
    ]

    print("\n" + "=" * 70)
    print("DOCUMENT-LEVEL TRAIN / TEST SPLIT")
    print("=" * 70)

    print(
        f"\nTotal documents:  {len(document_ids)}"
    )

    print(
        f"Training documents: {len(train_documents)}"
    )

    print(
        f"Testing documents:  {len(test_documents)}"
    )

    print(
        f"\nTraining passages: {len(train_records)}"
    )

    print(
        f"Testing passages:  {len(test_records)}"
    )

    # Safety check
    overlap = train_documents.intersection(
        test_documents
    )

    if overlap:

        raise RuntimeError(
            "ERROR: Document leakage detected!"
        )

    print(
        "\n✓ No document overlap between train and test"
    )

    return train_records, test_records


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(records, model):

    texts = [
        record["text"]
        for record in records
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return np.asarray(embeddings)


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

def train_classifier(X_train, y_train):

    print("\n" + "=" * 70)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 70)

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    classifier.fit(
        X_train,
        y_train
    )

    print("\n✓ Classifier trained")

    return classifier


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    classifier,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)

    predictions = classifier.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print(
        f"\nAccuracy:    {accuracy:.4f}"
    )

    print(
        f"Macro-F1:    {macro_f1:.4f}"
    )

    print(
        f"Weighted-F1: {weighted_f1:.4f}"
    )

    print("\n" + "-" * 70)
    print("PER-CLASS RESULTS")
    print("-" * 70)

    report = classification_report(
        y_test,
        predictions,
        digits=4
    )

    print(report)

    return (
        predictions,
        accuracy,
        macro_f1,
        weighted_f1,
        report,
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def show_confusion_matrix(
    y_test,
    predictions
):

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=MAJOR_LABELS
    )

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print("\nRows = actual")
    print("Columns = predicted\n")

    short_names = [
        "Subsumtion",
        "PreviousCaseLaw",
        "Proportionality",
        "Decision",
    ]

    print(
        f"{'Actual':22}",
        end=""
    )

    for name in short_names:

        print(
            f"{name:18}",
            end=""
        )

    print()

    for i, name in enumerate(short_names):

        print(
            f"{name:22}",
            end=""
        )

        for j in range(len(short_names)):

            print(
                f"{matrix[i][j]:18}",
                end=""
            )

        print()

    return matrix


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    accuracy,
    macro_f1,
    weighted_f1,
    matrix,
    train_documents,
    test_documents,
):

    results = {

        "experiment":
            "Gap 2 document-level classification",

        "model":
            MODEL_NAME,

        "classifier":
            "LogisticRegression",

        "labels":
            MAJOR_LABELS,

        "samples_per_label":
            SAMPLES_PER_LABEL,

        "test_size":
            TEST_SIZE,

        "random_state":
            RANDOM_STATE,

        "split_strategy":
            "document-level",

        "train_documents":
            train_documents,

        "test_documents":
            test_documents,

        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),

        "weighted_f1":
            float(weighted_f1),

        "confusion_matrix":
            matrix.tolist(),
    }

    output_path = (
        OUTPUT_DIR /
        "gap2_document_level_classifier_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nResults saved to:\n{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "RMU:ECHR GAP 2 — DOCUMENT-LEVEL CLASSIFICATION"
    )
    print("=" * 70)

    dataset = load_dataset()

    print(
        f"\nTotal dataset records: {len(dataset)}"
    )

    records = select_samples(
        dataset
    )

    # --------------------------------------------------------
    # DOCUMENT-LEVEL SPLIT
    # --------------------------------------------------------

    train_records, test_records = split_by_document(
        records
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING EMBEDDING MODEL")
    print("=" * 70)

    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    print("\nGenerating training embeddings...")

    X_train = generate_embeddings(
        train_records,
        model
    )

    print(
        f"Training embedding matrix: {X_train.shape}"
    )

    print("\nGenerating testing embeddings...")

    X_test = generate_embeddings(
        test_records,
        model
    )

    print(
        f"Testing embedding matrix: {X_test.shape}"
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    y_train = np.array([
        record["argument_type"]
        for record in train_records
    ])

    y_test = np.array([
        record["argument_type"]
        for record in test_records
    ])

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    classifier = train_classifier(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    (
        predictions,
        accuracy,
        macro_f1,
        weighted_f1,
        report,
    ) = evaluate(
        classifier,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    matrix = show_confusion_matrix(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    train_documents = len(
        set(
            record["document_id"]
            for record in train_records
        )
    )

    test_documents = len(
        set(
            record["document_id"]
            for record in test_records
        )
    )

    save_results(
        accuracy,
        macro_f1,
        weighted_f1,
        matrix,
        train_documents,
        test_documents,
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DOCUMENT-LEVEL EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()