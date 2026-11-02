"""Source-grounded hierarchical summarization for uploaded judgments."""

import math
import logging
import os
import random
import re
from collections import Counter
from functools import lru_cache

import numpy as np


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "his", "in",
    "into", "is", "it", "its", "of", "on", "or", "our", "she", "that",
    "the", "their", "them", "there", "these", "they", "this", "those",
    "to", "was", "were", "which", "who", "will", "with", "would",
}
_LEGAL_MARKERS = (
    "held", "we hold", "court held", "appeal", "petition", "dismissed",
    "allowed", "ordered", "judgment", "conclusion", "therefore", "section",
    "article", "provision", "constitutional", "finding", "issue", "question",
    "submission", "contended", "prosecution", "evidence", "reasoning",
)
logger = logging.getLogger("legal_judgment_summarizer.summarization")
_GENERATION_INSTRUCTION = (
    "Summarize only statements supported by the source. Do not add facts, arguments, "
    "citations, statutory provisions, dates, or holdings. Preserve party positions "
    "and legal outcomes only when explicitly stated.\nSOURCE PASSAGES:\n"
)
_COHERENCE_RANDOM_SEED = 42
_COHERENCE_PERMUTATIONS = 5


@lru_cache(maxsize=1)
def _load_generator(model_name):
    from transformers import pipeline

    return pipeline("summarization", model=model_name)


def _tokens(text):
    return re.findall(r"[A-Za-z][A-Za-z'-]{2,}|\d+(?:\([\w]+\))?", text.lower())


def _sentences(text):
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _source_reference(passage):
    position = passage["original_position"]
    return {
        "passage_id": passage["passage_id"],
        "start_char": position["start_char"],
        "end_char": position["end_char"],
    }


def _select_passage_groups(passages):
    windows = [passages[start:start + 4] for start in range(0, len(passages), 4)]
    if len(windows) <= 6:
        return windows

    target_passages = min(len(passages), min(96, max(24, math.ceil(len(passages) * 0.18))))
    target_windows = max(1, math.ceil(target_passages / 4))

    def window_score(window_index):
        window = windows[window_index]
        text = " ".join(passage["text"].lower() for passage in window)
        marker_score = sum(marker in text for marker in _LEGAL_MARKERS)
        boundary_score = 1.5 if window_index == 0 or window_index == len(windows) - 1 else 0
        return marker_score + boundary_score

    ranked = sorted(range(len(windows)), key=lambda index: (-window_score(index), index))
    selected = []
    used_passages = 0
    for window_index in ranked:
        if any(abs(window_index - chosen) <= 1 for chosen in selected):
            continue
        selected.append(window_index)
        used_passages += len(windows[window_index])
        if used_passages >= target_passages or len(selected) >= target_windows:
            break

    return [windows[index] for index in sorted(selected)]


def _ground_generated_claims(text, source_text):
    source_tokens = set(_tokens(source_text))
    accepted = []
    unsupported = []

    for sentence in _sentences(text):
        claim_tokens = [token for token in _tokens(sentence) if token not in _STOPWORDS]
        if not claim_tokens:
            unsupported.append({"claim": sentence, "reason": "No verifiable content tokens."})
            continue

        overlap = sum(token in source_tokens for token in claim_tokens) / len(claim_tokens)
        identifiers = re.findall(r"\b\d+(?:\([A-Za-z0-9]+\))?(?:-[A-Za-z0-9]+)?\b", sentence)
        missing_identifiers = [
            identifier for identifier in identifiers
            if identifier.lower() not in source_text.lower()
        ]
        if overlap < 0.45 or missing_identifiers:
            reason = "Source token overlap below threshold." if overlap < 0.45 else "Identifier absent from selected sources."
            unsupported.append({
                "claim": sentence,
                "reason": reason,
                "source_token_overlap": round(overlap, 4),
            })
            continue
        accepted.append(sentence)

    return " ".join(accepted), unsupported


class HierarchicalSummarizer:
    def __init__(self, generator=None, embedding_analyzer=None):
        from backend.app.services.passage_builder import PassageBuilder
        from backend.app.services.semantic_analyzer import SemanticAnalyzer

        self._generator = generator
        self.embedding_analyzer = embedding_analyzer or SemanticAnalyzer()
        self.passage_builder = PassageBuilder(sentences_per_passage=5)

    @property
    def generator(self):
        if self._generator is None:
            model_name = os.getenv("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
            self._generator = _load_generator(model_name)
        return self._generator

    def _generate(self, source_text):
        result = self.generator(
            _GENERATION_INSTRUCTION + source_text,
            max_length=220,
            min_length=20,
            do_sample=False,
            no_repeat_ngram_size=3,
            truncation=True,
        )
        return result[0]["summary_text"].strip()

    def summarize(self, text, document_id="uploaded-judgment", extraction_metadata=None):
        from backend.app.services.rmu_classifier import predict_transferred_functions

        passages = self.passage_builder.build_passages(text, document_id=document_id)
        if not passages:
            raise ValueError("No passages could be built from the extracted document text.")

        encoded = self.embedding_analyzer.encode_passages(passages, batch_size=32)
        passages = encoded["passages"]
        logger.info("Built %d source passages for %s", len(passages), document_id)
        group_sources = _select_passage_groups(passages)
        selected_passages = [passage for group in group_sources for passage in group]
        logger.info(
            "Selected %d passages in %d source groups for %s",
            len(selected_passages), len(group_sources), document_id,
        )

        transfer_result = predict_transferred_functions(
            passages,
            self.embedding_analyzer,
        )
        predictions_by_id = {}
        if transfer_result:
            predictions_by_id = {
                int(passage_id): label
                for passage_id, label in zip(
                    transfer_result["passage_ids"], transfer_result["predictions"]
                )
            }

        intermediate = []
        generation_trace = []
        unsupported_claims = []
        for group_index, group in enumerate(group_sources):
            group_text = "\n".join(passage["text"] for passage in group)
            references = [_source_reference(passage) for passage in group]
            candidate = self._generate(group_text)
            grounded, unsupported = _ground_generated_claims(candidate, group_text)
            unsupported_claims.extend(unsupported)
            item = {
                "level": 1,
                "group_index": group_index,
                "summary": grounded,
                "source_references": references,
                "_source_text": group_text,
            }
            intermediate.append(item)
            generation_trace.append({key: value for key, value in item.items() if not key.startswith("_")})
            if (group_index + 1) % 10 == 0 or group_index + 1 == len(group_sources):
                logger.info(
                    "Generated level 1 summaries: %d/%d for %s",
                    group_index + 1, len(group_sources), document_id,
                )

        # Reduce summaries in bounded batches so long judgments never become one prompt.
        current_level = intermediate
        generation_level = 2
        while len(current_level) > 4:
            next_level = []
            for start in range(0, len(current_level), 4):
                children = current_level[start:start + 4]
                child_text = "\n".join(
                    f"Sources {child['source_references']}: {child['summary']}"
                    for child in children if child["summary"]
                )
                source_text = "\n".join(child["_source_text"] for child in children)
                references = [
                    reference for child in children
                    for reference in child["source_references"]
                ]
                candidate = self._generate(child_text) if child_text else ""
                grounded, unsupported = _ground_generated_claims(candidate, source_text)
                unsupported_claims.extend(unsupported)
                item = {
                    "level": generation_level,
                    "group_index": len(next_level),
                    "summary": grounded,
                    "source_references": references,
                    "_source_text": source_text,
                }
                next_level.append(item)
                generation_trace.append({key: value for key, value in item.items() if not key.startswith("_")})
            current_level = next_level
            generation_level += 1

        final_input = "\n".join(
            f"Source references {item['source_references']}: {item['summary']}"
            for item in current_level if item["summary"]
        )
        final_source = "\n".join(item["_source_text"] for item in current_level)
        candidate = self._generate(final_input) if final_input else ""
        final_summary, final_unsupported = _ground_generated_claims(candidate, final_source)
        unsupported_claims.extend(final_unsupported)
        final_references = [
            reference for item in current_level
            for reference in item["source_references"]
        ]
        generation_trace.append({
            "level": generation_level,
            "group_index": 0,
            "summary": final_summary,
            "source_references": final_references,
        })

        if not final_summary:
            final_summary = "No generated claims passed the automatic source-grounding checks."
        logger.info(
            "Completed %s with %d selected passages and %d flagged claims",
            document_id, len(selected_passages), len(unsupported_claims),
        )

        adjacent_similarities = []
        adjacent_character_gaps = []
        for first, second in zip(passages, passages[1:]):
            first_embedding = np.asarray(first["embedding"])
            second_embedding = np.asarray(second["embedding"])
            denominator = np.linalg.norm(first_embedding) * np.linalg.norm(second_embedding)
            if denominator:
                adjacent_similarities.append(float(np.dot(first_embedding, second_embedding) / denominator))
            adjacent_character_gaps.append(max(
                second["original_position"]["start_char"]
                - first["original_position"]["end_char"],
                0,
            ))

        shuffled_coherence_means = []
        if len(passages) >= 2:
            rng = random.Random(_COHERENCE_RANDOM_SEED)
            original_order = list(range(len(passages)))
            for _ in range(_COHERENCE_PERMUTATIONS):
                shuffled_order = original_order.copy()
                for _attempt in range(10):
                    rng.shuffle(shuffled_order)
                    if shuffled_order != original_order:
                        break
                else:
                    shuffled_order[0], shuffled_order[1] = shuffled_order[1], shuffled_order[0]

                similarities = []
                for first_index, second_index in zip(shuffled_order, shuffled_order[1:]):
                    first_embedding = np.asarray(passages[first_index]["embedding"])
                    second_embedding = np.asarray(passages[second_index]["embedding"])
                    denominator = np.linalg.norm(first_embedding) * np.linalg.norm(second_embedding)
                    if denominator:
                        similarities.append(
                            float(np.dot(first_embedding, second_embedding) / denominator)
                        )
                if similarities:
                    shuffled_coherence_means.append(float(np.mean(similarities)))

        nearby_relationships = []
        for first_index, first in enumerate(passages):
            for second in passages[first_index + 1:first_index + 6]:
                first_embedding = np.asarray(first["embedding"])
                second_embedding = np.asarray(second["embedding"])
                denominator = np.linalg.norm(first_embedding) * np.linalg.norm(second_embedding)
                similarity = (
                    float(np.dot(first_embedding, second_embedding) / denominator)
                    if denominator else None
                )
                nearby_relationships.append({
                    "passage_i": first["passage_id"],
                    "passage_j": second["passage_id"],
                    "passage_distance": second["passage_id"] - first["passage_id"],
                    "character_gap": max(
                        second["original_position"]["start_char"]
                        - first["original_position"]["end_char"],
                        0,
                    ),
                    "semantic_similarity": similarity,
                })
        related_pairs = sorted(
            (pair for pair in nearby_relationships if pair["semantic_similarity"] is not None),
            key=lambda pair: pair["semantic_similarity"],
            reverse=True,
        )[:20]

        total_characters = max(len(text), 1)
        selected_characters = sum(
            reference["end_char"] - reference["start_char"]
            for reference in [_source_reference(passage) for passage in selected_passages]
        )
        selected_ids = {passage["passage_id"] for passage in selected_passages}
        selected_labels = Counter(
            predictions_by_id[passage_id]
            for passage_id in selected_ids if passage_id in predictions_by_id
        )
        all_labels = Counter(transfer_result["predictions"]) if transfer_result else Counter()
        functional_coverage = {
            label: {
                "selected_count": count,
                "document_count": all_labels[label],
                "selected_fraction": round(count / all_labels[label], 4) if all_labels[label] else None,
            }
            for label, count in selected_labels.items()
        }

        return {
            "summary": final_summary,
            "metadata": {
                "document_id": document_id,
                "passage_count": len(passages),
                "selected_passage_count": len(selected_passages),
                "selected_passage_indices": sorted(selected_ids),
                "source_references": [
                    _source_reference(passage) for passage in selected_passages
                ],
                "generation_trace": generation_trace,
                "content_coverage_fraction": round(
                    min(selected_characters / total_characters, 1.0), 4
                ),
                "functional_coverage": functional_coverage,
                "functional_predictions": {
                    "provenance": "Transferred RMU:ECHR functional predictions; not Indian ground truth.",
                    "validated_on_indian_data": False,
                    "predictions": [
                        item for item in transfer_result["items"]
                        if item["passage_id"] in selected_ids
                    ] if transfer_result else [],
                    "available": bool(transfer_result),
                },
                "relationship_analysis": {
                    "passage_order_preserved": True,
                    "adjacent_pair_count": len(adjacent_similarities),
                    "mean_adjacent_character_gap": (
                        round(float(np.mean(adjacent_character_gaps)), 2)
                        if adjacent_character_gaps else None
                    ),
                    "mean_adjacent_semantic_similarity": (
                        round(float(np.mean(adjacent_similarities)), 4)
                        if adjacent_similarities else None
                    ),
                    "coherence_signal_status": "Exploratory; not validated on Indian judgments.",
                    "local_pair_count": len(nearby_relationships),
                    "highest_similarity_local_pairs": related_pairs,
                    "actor_statistics_imported": False,
                },
                "coherence_analysis": {
                    "ordered_adjacent_mean_similarity": (
                        round(float(np.mean(adjacent_similarities)), 4)
                        if adjacent_similarities else None
                    ),
                    "shuffled_adjacent_mean_similarity": (
                        round(float(np.mean(shuffled_coherence_means)), 4)
                        if shuffled_coherence_means else None
                    ),
                    "ordered_minus_shuffled": (
                        round(
                            float(np.mean(adjacent_similarities) - np.mean(shuffled_coherence_means)),
                            4,
                        )
                        if adjacent_similarities and shuffled_coherence_means else None
                    ),
                    "permutation_count": len(shuffled_coherence_means),
                    "random_seed": _COHERENCE_RANDOM_SEED,
                    "status": "Exploratory; this comparison is not validated on Indian judgments.",
                },
                "grounding": {
                    "method": "Lexical source overlap and identifier-presence checks; not entailment verification.",
                    "unsupported_claim_count": len(unsupported_claims),
                    "unsupported_claims": unsupported_claims,
                    "limitations": [
                        "Automatic overlap checks cannot establish legal entailment or detect every contradiction.",
                        "Abstractive model output requires human legal review before reliance.",
                    ],
                },
                "generation_levels": generation_level,
                "extraction": extraction_metadata or {},
                "signals": {
                    "indian_validation": [
                        "Gap 1 passage-order/source-offset checks were exercised on the Vijay Madanlal judgment artifact; this does not validate legal-function accuracy."
                    ],
                    "exploratory_transfers": [
                        "RMU:ECHR functional label predictions",
                        "Adjacent semantic similarity as a coherence proxy",
                    ],
                },
            },
        }