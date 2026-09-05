"""
Helper module to encapsulate Gap 2A classifier training and prediction.

This module preserves the exact behavior of test_embeddings_classifier.py
while allowing the classifier to be reused by other experiments (e.g., Gap 2B).

CRITICAL: This helper must NOT alter the existing Gap 2A behavior or results.
"""

from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# ============================================================
# GAP 2A CONFIGURATION (IMMUTABLE)
# ============================================================

class Gap2AConfig:
    """Configuration for Gap 2A RMU:ECHR experiment."""
    
    DATASET_PATH = Path("data/experiments/gap2/rmu_echr_annotations.json")
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
# TRAIN GAP 2A CLASSIFIER
# ============================================================

def load_rmu_echr_dataset():
    """Load the RMU:ECHR dataset (Gap 2A data)."""
    with open(Gap2AConfig.DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def select_gap2a_samples(dataset):
    """Select samples from RMU:ECHR dataset for training."""
    selected = []
    
    for label in Gap2AConfig.MAJOR_LABELS:
        examples = [
            record
            for record in dataset
            if record["argument_type"] == label
        ]
        examples = examples[:Gap2AConfig.SAMPLES_PER_LABEL]
        selected.extend(examples)
    
    return selected


def split_by_document_gap2a(records):
    """Split records by document (document-level split)."""
    document_ids = sorted(
        set(record["document_id"] for record in records)
    )
    
    train_documents, test_documents = train_test_split(
        document_ids,
        test_size=Gap2AConfig.TEST_SIZE,
        random_state=Gap2AConfig.RANDOM_STATE
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
    
    return train_records, test_records


def generate_embeddings(records, model, batch_size=32, show_progress=True):
    """Generate embeddings for a list of records."""
    texts = [record["text"] for record in records]
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True
    )
    
    return np.asarray(embeddings)


def train_gap2a_classifier(X_train, y_train):
    """Train the Gap 2A Logistic Regression classifier."""
    classifier = LogisticRegression(
        max_iter=2000,
        random_state=Gap2AConfig.RANDOM_STATE
    )
    classifier.fit(X_train, y_train)
    return classifier


def create_label_encoder():
    """Create a label encoder for Gap 2A major labels."""
    le = LabelEncoder()
    le.fit(Gap2AConfig.MAJOR_LABELS)
    return le


# ============================================================
# TRAIN AND RETURN FULL CLASSIFIER PACKAGE
# ============================================================

def train_full_gap2a_classifier(verbose=True):
    """
    Train a full Gap 2A classifier from RMU:ECHR data.
    
    Returns:
        dict: Dictionary containing:
            - 'classifier': trained LogisticRegression
            - 'label_encoder': LabelEncoder for the 4 major labels
            - 'model': SentenceTransformer model
            - 'config': Gap2AConfig
            - 'train_records': training passages
            - 'test_records': test passages
    """
    if verbose:
        print("=" * 70)
        print("TRAINING GAP 2A CLASSIFIER FOR REUSE")
        print("=" * 70)
    
    # Load and prepare data
    if verbose:
        print("\n[1/5] Loading RMU:ECHR dataset...")
    dataset = load_rmu_echr_dataset()
    records = select_gap2a_samples(dataset)
    
    if verbose:
        print(f"       Selected {len(records)} passages")
    
    train_records, test_records = split_by_document_gap2a(records)
    
    if verbose:
        print(f"       Train passages: {len(train_records)}")
        print(f"       Test passages: {len(test_records)}")
    
    # Load embedding model
    if verbose:
        print("\n[2/5] Loading SentenceTransformer model...")
    model = SentenceTransformer(Gap2AConfig.MODEL_NAME)
    
    if verbose:
        print(f"       Model: {Gap2AConfig.MODEL_NAME}")
    
    # Generate embeddings
    if verbose:
        print("\n[3/5] Generating training embeddings...")
    X_train = generate_embeddings(train_records, model, show_progress=verbose)
    
    if verbose:
        print(f"       Shape: {X_train.shape}")
    
    if verbose:
        print("\n[4/5] Generating test embeddings...")
    X_test = generate_embeddings(test_records, model, show_progress=verbose)
    
    # Train classifier
    if verbose:
        print("\n[5/5] Training LogisticRegression classifier...")
    
    y_train = np.array([record["argument_type"] for record in train_records])
    classifier = train_gap2a_classifier(X_train, y_train)
    
    if verbose:
        print("       ✓ Classifier trained")
    
    # Label encoder
    label_encoder = create_label_encoder()
    
    if verbose:
        print("\n" + "=" * 70)
        print("CLASSIFIER READY FOR REUSE")
        print("=" * 70)
    
    return {
        'classifier': classifier,
        'label_encoder': label_encoder,
        'model': model,
        'config': Gap2AConfig,
        'train_records': train_records,
        'test_records': test_records,
        'X_train': X_train,
        'X_test': X_test,
    }


# ============================================================
# MAKE PREDICTIONS ON NEW DATA
# ============================================================

def predict_on_passages(classifier, passages, embedding_model, batch_size=32):
    """
    Generate predictions for a list of passages using a trained classifier.
    
    Args:
        classifier: trained LogisticRegression
        passages: list of passage dicts (must have 'text' key)
        embedding_model: SentenceTransformer model
        batch_size: embedding batch size
        
    Returns:
        dict: Contains:
            - 'embeddings': array of embeddings (shape: [num_passages, 384])
            - 'predictions': array of predicted labels
            - 'probabilities': array of confidence scores (shape: [num_passages, num_classes])
            - 'major_labels': list of major labels in classifier order
    """
    # Extract texts
    texts = [p['text'] for p in passages]
    
    # Generate embeddings
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    embeddings = np.asarray(embeddings)
    
    # Make predictions
    predictions = classifier.predict(embeddings)
    
    # Get prediction probabilities
    probabilities = classifier.predict_proba(embeddings)
    
    return {
        'embeddings': embeddings,
        'predictions': predictions,
        'probabilities': probabilities,
        'major_labels': list(Gap2AConfig.MAJOR_LABELS),
    }
