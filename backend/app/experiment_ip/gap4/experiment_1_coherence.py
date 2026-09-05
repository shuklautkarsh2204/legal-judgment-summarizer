"""
Gap 4 Experiment 1: Discourse coherence and logical progression.

This is an exploratory control-condition experiment. It compares adjacent
signals in the naturally ordered original RMU:ECHR documents with the same
signals after within-document random permutations. It does not train a
coherence classifier or claim legal correctness.
"""

import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

# Make the repository root importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).parents[4]))

from backend.app.experiment_ip.gap3.test_passage_relationships import (
    Gap3Analyzer,
    load_document_texts,
)

RANDOM_SEED = 42
PERMUTATIONS_PER_DOCUMENT = 5
MIN_PASSAGES_PER_DOCUMENT = 2
EMBEDDING_DIMENSION = 384
MAX_SHUFFLE_ATTEMPTS = 10


def cosine_similarity(first, second):
    first = np.asarray(first)
    second = np.asarray(second)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        raise ValueError("Cannot calculate similarity for a zero-norm embedding")
    return float(np.dot(first, second) / denominator)


def transition_key(value):
    return value if value is not None else "UNKNOWN"


def passage_signature(passage):
    return {
        "document_id": passage.document_id,
        "passage_index": passage.passage_index,
        "text": passage.text,
        "begin_char": passage.begin_char,
        "end_char": passage.end_char,
        "embedding_dimension": int(np.asarray(passage.embedding).shape[0]),
    }


def validate_sequence(original, candidate, document_id):
    if len(original) != len(candidate):
        raise ValueError(f"Passage count changed for {document_id}")

    original_indices = [passage.passage_index for passage in original]
    candidate_indices = [passage.passage_index for passage in candidate]
    if sorted(original_indices) != sorted(candidate_indices):
        raise ValueError(f"Passage identity changed for {document_id}")
    if len(set(candidate_indices)) != len(candidate_indices):
        raise ValueError(f"Duplicate passage in shuffled sequence for {document_id}")

    for source, shuffled in zip(sorted(original, key=lambda item: item.passage_index),
                                sorted(candidate, key=lambda item: item.passage_index)):
        if source.document_id != document_id or shuffled.document_id != document_id:
            raise ValueError(f"Document membership changed for {document_id}")
        if passage_signature(source) != passage_signature(shuffled):
            raise ValueError(f"Passage metadata changed for {document_id}")
        if not np.array_equal(source.embedding, shuffled.embedding):
            raise ValueError(f"Embedding changed for {document_id}")


def sequence_metrics(sequence):
    semantic_scores = []
    actor_overlap = []
    argument_transitions = Counter()
    actor_transitions = Counter()

    for first, second in zip(sequence, sequence[1:]):
        semantic_scores.append(cosine_similarity(first.embedding, second.embedding))
        first_actors = set(first.get_actors())
        second_actors = set(second.get_actors())
        if first_actors and second_actors:
            actor_overlap.append(bool(first_actors & second_actors))
        else:
            actor_overlap.append(None)

        argument_transitions[(
            transition_key(first.primary_argument_type()),
            transition_key(second.primary_argument_type()),
        )] += 1
        actor_transitions[(
            transition_key(first.primary_actor()),
            transition_key(second.primary_actor()),
        )] += 1

    known_actor_values = [value for value in actor_overlap if value is not None]
    return {
        "semantic_scores": semantic_scores,
        "semantic_mean": float(np.mean(semantic_scores)) if semantic_scores else None,
        "semantic_median": float(np.median(semantic_scores)) if semantic_scores else None,
        "semantic_std": float(np.std(semantic_scores)) if semantic_scores else None,
        "actor_overlap_count": int(sum(known_actor_values)),
        "actor_overlap_known_pairs": len(known_actor_values),
        "actor_overlap_proportion": (
            float(np.mean(known_actor_values)) if known_actor_values else None
        ),
        "argument_transitions": argument_transitions,
        "actor_transitions": actor_transitions,
        "adjacent_pair_count": len(sequence) - 1,
    }


def aggregate_scores(values):
    values = [value for value in values if value is not None]
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "std": float(np.std(values)) if values else None,
    }


def serialise_counter(counter):
    return [
        {"from": source, "to": target, "count": count}
        for (source, target), count in sorted(
            counter.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def write_plot(output_dir, original_scores, shuffled_scores, document_means):
    plt.figure(figsize=(9, 5))
    plt.hist(original_scores, bins=30, alpha=0.65, label="Original", density=True)
    plt.hist(shuffled_scores, bins=30, alpha=0.65, label="Shuffled", density=True)
    plt.title("Adjacent-passage semantic continuity")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "semantic_continuity_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    original = [row["original_mean"] for row in document_means]
    shuffled = [row["shuffled_mean"] for row in document_means]
    positions = np.arange(len(document_means))
    width = 0.4
    plt.scatter(positions - width / 2, original, label="Original", s=12)
    plt.scatter(positions + width / 2, shuffled, label="Shuffled mean", s=12)
    plt.title("Document-level mean semantic continuity")
    plt.xlabel("Eligible document (deterministic order)")
    plt.ylabel("Mean adjacent cosine similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "document_level_semantic_continuity.png", dpi=150)
    plt.close()


def write_transition_heatmap(output_dir, transitions, title, filename):
    labels = sorted({label for pair in transitions for label in pair})
    matrix = np.zeros((len(labels), len(labels)))
    label_index = {label: index for index, label in enumerate(labels)}
    for (source, target), count in transitions.items():
        matrix[label_index[source], label_index[target]] = count

    plt.figure(figsize=(max(7, len(labels) * 0.45), max(6, len(labels) * 0.4)))
    plt.imshow(matrix, aspect="auto", cmap="Blues")
    plt.colorbar(label="Transition count")
    plt.xticks(range(len(labels)), labels, rotation=65, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Next passage label")
    plt.ylabel("Current passage label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def main():
    project_root = Path(__file__).parents[4]
    annotation_path = project_root / "data" / "experiments" / "gap2" / "rmu_echr_annotations.json"
    output_dir = project_root / "data" / "experiments" / "gap4"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("GAP 4 EXPERIMENT 1: ORIGINAL VS WITHIN-DOCUMENT DISRUPTED SEQUENCES")
    print("Using original RMU:ECHR documents and Gap 3 passage construction")

    analyzer = Gap3Analyzer(random_seed=RANDOM_SEED)
    analyzer.load_annotations(annotation_path)
    document_texts, loader, _ = load_document_texts(annotation_path)
    if not document_texts:
        raise RuntimeError("No original RMU:ECHR documents were loaded")
    if not analyzer.validate_annotations_and_documents(document_texts, loader):
        print("WARNING: Gap 3 validation reported no valid offsets; continuing is unsafe")
        raise RuntimeError("Original annotation validation failed")

    analyzer.build_passages_for_docs(document_texts)
    eligible = {
        document_id: passages
        for document_id, passages in sorted(analyzer.passages_by_doc.items())
        if len(passages) >= MIN_PASSAGES_PER_DOCUMENT
    }
    if not eligible:
        raise RuntimeError("No documents contain enough passages")

    all_original_scores = []
    all_shuffled_scores = []
    original_argument_transitions = Counter()
    shuffled_argument_transitions = Counter()
    original_actor_transitions = Counter()
    shuffled_actor_transitions = Counter()
    document_means = []
    raw_sequences = []
    rng = random.Random(RANDOM_SEED)

    print("Starting original-vs-shuffled sequence analysis...")
    total_documents = len(eligible)
    for document_index, (document_id, passages) in enumerate(eligible.items(), start=1):
        expected_indices = list(range(len(passages)))
        actual_indices = [passage.passage_index for passage in passages]
        if actual_indices != expected_indices:
            raise ValueError(f"Original passage ordering is not sequential for {document_id}")
        for passage in passages:
            if np.asarray(passage.embedding).shape != (EMBEDDING_DIMENSION,):
                raise ValueError(f"Unexpected embedding dimension for {document_id}")

        original_metrics = sequence_metrics(passages)
        all_original_scores.extend(original_metrics["semantic_scores"])
        original_argument_transitions.update(original_metrics["argument_transitions"])
        original_actor_transitions.update(original_metrics["actor_transitions"])

        shuffled_means = []
        shuffled_actor_values = []
        shuffled_records = []
        for permutation_index in range(PERMUTATIONS_PER_DOCUMENT):
            shuffled = list(passages)
            for _ in range(MAX_SHUFFLE_ATTEMPTS):
                rng.shuffle(shuffled)
                if [passage.passage_index for passage in shuffled] != expected_indices:
                    break
            else:
                shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
            validate_sequence(passages, shuffled, document_id)
            metrics = sequence_metrics(shuffled)
            all_shuffled_scores.extend(metrics["semantic_scores"])
            shuffled_means.append(metrics["semantic_mean"])
            if metrics["actor_overlap_proportion"] is not None:
                shuffled_actor_values.append(metrics["actor_overlap_proportion"])
            shuffled_actor_transitions.update(metrics["actor_transitions"])
            shuffled_argument_transitions.update(metrics["argument_transitions"])
            shuffled_records.append({
                "permutation_index": permutation_index,
                "passage_indices": [passage.passage_index for passage in shuffled],
                "metrics": {
                    key: value for key, value in metrics.items()
                    if key not in {"semantic_scores", "argument_transitions", "actor_transitions"}
                },
            })
        document_means.append({
            "document_id": document_id,
            "passage_count": len(passages),
            "original_mean": original_metrics["semantic_mean"],
            "shuffled_mean": float(np.mean(shuffled_means)),
            "original_actor_overlap_proportion": original_metrics["actor_overlap_proportion"],
            "shuffled_actor_overlap_proportion": (
                float(np.mean(shuffled_actor_values)) if shuffled_actor_values else None
            ),
        })
        raw_sequences.append({
            "document_id": document_id,
            "passage_indices": expected_indices,
            "shuffled_sequences": shuffled_records,
        })
        if document_index % 25 == 0 or document_index == total_documents:
            print(
                f"Gap 4 progress: {document_index}/{total_documents} documents processed"
            )

    paired_original = [row["original_mean"] for row in document_means]
    paired_shuffled = [row["shuffled_mean"] for row in document_means]
    if len(paired_original) >= 2:
        test = wilcoxon(paired_original, paired_shuffled, alternative="two-sided")
        statistical_test = {
            "name": "Wilcoxon signed-rank test",
            "unit": "document-level mean semantic continuity",
            "n_documents": len(paired_original),
            "statistic": float(test.statistic),
            "p_value": float(test.pvalue),
        }
    else:
        statistical_test = {
            "name": "Wilcoxon signed-rank test",
            "unit": "document-level mean semantic continuity",
            "n_documents": len(paired_original),
            "statistic": None,
            "p_value": None,
            "reason": "At least two eligible documents are required",
        }

    original_actor_mean = aggregate_scores([
        row["original_actor_overlap_proportion"] for row in document_means
    ])
    shuffled_actor_mean = aggregate_scores([
        row["shuffled_actor_overlap_proportion"] for row in document_means
    ])
    aggregate = {
        "documents_used": len(eligible),
        "passages_used": sum(len(passages) for passages in eligible.values()),
        "original_adjacent_pairs": len(all_original_scores),
        "shuffled_sequences": len(eligible) * PERMUTATIONS_PER_DOCUMENT,
        "original_semantic_continuity": aggregate_scores(all_original_scores),
        "shuffled_semantic_continuity": aggregate_scores(all_shuffled_scores),
        "semantic_mean_difference": float(np.mean(all_original_scores) - np.mean(all_shuffled_scores)),
        "original_document_level_semantic_means": aggregate_scores(paired_original),
        "shuffled_document_level_semantic_means": aggregate_scores(paired_shuffled),
        "actor_continuity_original_document_means": original_actor_mean,
        "actor_continuity_shuffled_document_means": shuffled_actor_mean,
        "actor_continuity_mean_difference": (
            original_actor_mean["mean"] - shuffled_actor_mean["mean"]
            if original_actor_mean["mean"] is not None and shuffled_actor_mean["mean"] is not None
            else None
        ),
        "statistical_comparison": statistical_test,
    }

    results = {
        "metadata": {
            "experiment": "Gap 4 Experiment 1 - Discourse Coherence Control Comparison",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "random_seed": RANDOM_SEED,
            "permutations_per_document": PERMUTATIONS_PER_DOCUMENT,
            "sentences_per_passage": 5,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "coordinate_system": "Original RMU:ECHR document character offsets",
            "synthetic_text_created": False,
            "statistical_unit": "document-level aggregates for paired semantic comparison",
        },
        "validation": {
            "original_order_checked": True,
            "same_passages_after_shuffle_checked": True,
            "no_passage_loss_or_duplication_checked": True,
            "document_membership_preserved_checked": True,
            "original_coordinates_preserved_checked": True,
            "embedding_dimensions_checked": True,
            "labels_attached_to_passages_checked": True,
            "annotation_validation": analyzer.validation_report,
        },
        "aggregate_statistics": aggregate,
        "original_argument_type_transitions": serialise_counter(original_argument_transitions),
        "shuffled_argument_type_transitions": serialise_counter(shuffled_argument_transitions),
        "original_actor_transitions": serialise_counter(original_actor_transitions),
        "shuffled_actor_transitions": serialise_counter(shuffled_actor_transitions),
        "limitations": [
            "This experiment measures continuity signals, not legal correctness or full coherence.",
            "RMU:ECHR labels are related-domain legal-functional annotations, not Indian Supreme Court relationship gold labels.",
            "Random permutations are a control condition, not necessarily legally incorrect reasoning.",
            "Passage-level observations are summarized by document for semantic inference; transition counts remain descriptive.",
            "No trained coherence classifier, tuned threshold, or invented gold label is used.",
        ],
    }

    (output_dir / "gap4_experiment_1_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (output_dir / "gap4_experiment_1_raw_sequences.json").write_text(
        json.dumps(raw_sequences, indent=2), encoding="utf-8"
    )
    (output_dir / "gap4_experiment_1_configuration.json").write_text(
        json.dumps(results["metadata"], indent=2), encoding="utf-8"
    )
    with (output_dir / "gap4_document_level_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=document_means[0].keys())
        writer.writeheader()
        writer.writerows(document_means)

    write_plot(output_dir, all_original_scores, all_shuffled_scores, document_means)
    write_transition_heatmap(
        output_dir, original_argument_transitions,
        "Original-sequence argument-type transitions",
        "original_argument_type_transition_heatmap.png",
    )

    report = [
        "# Gap 4 Experiment 1 Report",
        "",
        "This exploratory experiment compares naturally ordered adjacent passages with multiple within-document random permutations.",
        "",
        f"- Documents used: {aggregate['documents_used']}",
        f"- Passages used: {aggregate['passages_used']}",
        f"- Original adjacent pairs: {aggregate['original_adjacent_pairs']}",
        f"- Shuffled sequences: {aggregate['shuffled_sequences']}",
        f"- Original semantic mean / median: {aggregate['original_semantic_continuity']['mean']:.6f} / {aggregate['original_semantic_continuity']['median']:.6f}",
        f"- Shuffled semantic mean / median: {aggregate['shuffled_semantic_continuity']['mean']:.6f} / {aggregate['shuffled_semantic_continuity']['median']:.6f}",
        f"- Semantic mean difference: {aggregate['semantic_mean_difference']:.6f}",
        f"- Actor-continuity mean difference: {aggregate['actor_continuity_mean_difference']}",
        f"- Statistical test: {statistical_test['name']} on {statistical_test['unit']} (p={statistical_test['p_value']})",
        f"- Distinguishability at alpha=0.05: {'yes' if statistical_test['p_value'] is not None and statistical_test['p_value'] < 0.05 else 'no or inconclusive'}",
        f"- Evidence for H4: {'present for the semantic document-level comparison' if statistical_test['p_value'] is not None and statistical_test['p_value'] < 0.05 else 'not established by the semantic document-level comparison'}",
        "",
        "## Interpretation",
        "",
        "The result characterizes measurable continuity signals under an order-preserving condition and a disrupted control condition. It does not prove legal coherence detection, legal validity, or causal discourse structure. Any evidence supporting H4 must be described cautiously after inspecting the generated values.",
        "",
        "Most common original argument-type transitions are available in `gap4_experiment_1_results.json`.",
    ]
    (output_dir / "GAP4_EXPERIMENT_1_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
