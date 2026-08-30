from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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

# Major classes with enough examples for meaningful comparison
MAJOR_LABELS = [
    "Subsumtion",
    "Vorherige Rechtsprechung des EGMR",
    "Verhältnismäßigkeitsprüfung – Angemessenheit",
    "Entscheidung des EGMR",
]

SAMPLES_PER_LABEL = 300


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# SELECT DATA
# ============================================================

def select_samples(dataset):

    selected = []

    for label in MAJOR_LABELS:

        examples = [
            record
            for record in dataset
            if record["argument_type"] == label
        ]

        # Fixed deterministic selection.
        # This makes the experiment reproducible.
        examples = examples[:SAMPLES_PER_LABEL]

        selected.extend(examples)

        print(
            f"{label}: "
            f"{len(examples)} samples selected"
        )

    return selected


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(records):

    print("\n" + "=" * 70)
    print("GENERATING EMBEDDINGS")
    print("=" * 70)

    model = SentenceTransformer(MODEL_NAME)

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

    print(
        f"\nEmbedding matrix shape: {embeddings.shape}"
    )

    return np.array(embeddings)


# ============================================================
# WITHIN / BETWEEN CLASS SIMILARITY
# ============================================================

def calculate_similarity(records, embeddings):

    labels = [
        record["argument_type"]
        for record in records
    ]

    similarity_matrix = cosine_similarity(embeddings)

    results = {}

    print("\n" + "=" * 70)
    print("SEMANTIC SIMILARITY ANALYSIS")
    print("=" * 70)

    unique_labels = MAJOR_LABELS

    for label_a in unique_labels:

        results[label_a] = {}

        indices_a = [
            i
            for i, label in enumerate(labels)
            if label == label_a
        ]

        for label_b in unique_labels:

            indices_b = [
                i
                for i, label in enumerate(labels)
                if label == label_b
            ]

            similarities = []

            for i in indices_a:

                for j in indices_b:

                    # Avoid comparing a passage with itself
                    if i == j:
                        continue

                    similarities.append(
                        similarity_matrix[i][j]
                    )

            if similarities:

                mean_similarity = np.mean(
                    similarities
                )

                results[label_a][label_b] = float(
                    mean_similarity
                )

            else:
                results[label_a][label_b] = None

    # Print matrix
    print("\nAverage cosine similarity:\n")

    print(
        f"{'':45}",
        end=""
    )

    for label in unique_labels:
        print(
            f"{label[:15]:>18}",
            end=""
        )

    print()

    for label_a in unique_labels:

        print(
            f"{label_a[:43]:45}",
            end=""
        )

        for label_b in unique_labels:

            value = results[label_a][label_b]

            if value is None:
                print(
                    f"{'N/A':>18}",
                    end=""
                )
            else:
                print(
                    f"{value:18.4f}",
                    end=""
                )

        print()

    return results


# ============================================================
# WITHIN VS BETWEEN CLASS SCORE
# ============================================================

def calculate_separation(records, embeddings):

    labels = np.array([
        record["argument_type"]
        for record in records
    ])

    similarity_matrix = cosine_similarity(
        embeddings
    )

    within_class = []
    between_class = []

    for i in range(len(labels)):

        for j in range(i + 1, len(labels)):

            similarity = similarity_matrix[i][j]

            if labels[i] == labels[j]:

                within_class.append(similarity)

            else:

                between_class.append(similarity)

    within_mean = np.mean(within_class)
    between_mean = np.mean(between_class)

    separation = within_mean - between_mean

    print("\n" + "=" * 70)
    print("WITHIN-CLASS VS BETWEEN-CLASS SIMILARITY")
    print("=" * 70)

    print(
        f"\nWithin-class similarity:   {within_mean:.4f}"
    )

    print(
        f"Between-class similarity:  {between_mean:.4f}"
    )

    print(
        f"Separation gap:             {separation:.4f}"
    )

    if separation > 0:

        print(
            "\n✓ Same-label passages are more "
            "semantically similar on average."
        )

    else:

        print(
            "\n⚠ Same-label passages are NOT "
            "more similar on average."
        )

    return {
        "within_class_similarity": float(within_mean),
        "between_class_similarity": float(between_mean),
        "separation_gap": float(separation),
    }


# ============================================================
# t-SNE VISUALIZATION
# ============================================================

def create_tsne(records, embeddings):

    print("\n" + "=" * 70)
    print("CREATING t-SNE VISUALIZATION")
    print("=" * 70)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca"
    )

    coordinates = tsne.fit_transform(
        embeddings
    )

    plt.figure(figsize=(12, 9))

    labels = [
        record["argument_type"]
        for record in records
    ]

    for label in MAJOR_LABELS:

        indices = [
            i
            for i, current_label in enumerate(labels)
            if current_label == label
        ]

        x = coordinates[indices, 0]
        y = coordinates[indices, 1]

        plt.scatter(
            x,
            y,
            label=label,
            alpha=0.6,
            s=20
        )

    plt.title(
        "RMU:ECHR Legal Argument Embedding Separation"
    )

    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")

    plt.legend()

    output_path = (
        OUTPUT_DIR /
        "gap2_embedding_tsne.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nVisualization saved to:\n{output_path}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(similarity_results, separation):

    output = {
        "model": MODEL_NAME,
        "labels": MAJOR_LABELS,
        "samples_per_label": SAMPLES_PER_LABEL,
        "similarity_matrix": similarity_results,
        "separation": separation,
    }

    output_path = (
        OUTPUT_DIR /
        "gap2_embedding_separation_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
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
    print("RMU:ECHR GAP 2 — EMBEDDING SEPARATION EXPERIMENT")
    print("=" * 70)

    dataset = load_dataset()

    print(
        f"\nDataset records: {len(dataset)}"
    )

    records = select_samples(dataset)

    embeddings = generate_embeddings(
        records
    )

    similarity_results = calculate_similarity(
        records,
        embeddings
    )

    separation = calculate_separation(
        records,
        embeddings
    )

    create_tsne(
        records,
        embeddings
    )

    save_results(
        similarity_results,
        separation
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()