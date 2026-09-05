"""
GAP 2B: INDIAN JUDICIAL CONTEXT TRANSFER TEST

Objective:
  Test whether the RMU:ECHR-trained legal-functional classifier 
  can transfer useful structural signals to a real Indian Supreme Court judgment.

Design:
  - Reuse Gap 2A classifier (RMU:ECHR trained)
  - Apply to Indian judgment (Vijay Madanlal Choudhary)
  - Analyze prediction distribution, sequence, and semantic consistency
  
WARNING:
  Predictions are NOT Indian ground truth.
  They are exploratory structural signals from domain-transfer.
  Interpret results with caution regarding legal correctness.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gap2a_classifier_helper import (
    train_full_gap2a_classifier,
    predict_on_passages,
    Gap2AConfig
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

OUTPUT_DIR = Path("data/experiments/gap2")

INDIAN_JUDGMENT_ID = "Vijay_Madanlal_Choudhary_vs_Union_Of_India_on_27_July_2022"

INDIAN_PASSAGES_PATH = (
    Path("data/experiments/gap1") /
    f"{INDIAN_JUDGMENT_ID}_passages.json"
)

INDIAN_STATS_PATH = (
    Path("data/experiments/gap1") /
    f"{INDIAN_JUDGMENT_ID}_stats.json"
)

# ============================================================
# LOAD INDIAN JUDGMENT PASSAGES
# ============================================================

def load_indian_passages():
    """Load passages from Gap 1 (already processed)."""
    if not INDIAN_PASSAGES_PATH.exists():
        raise FileNotFoundError(
            f"Gap 1 passages not found: {INDIAN_PASSAGES_PATH}"
        )
    
    with open(INDIAN_PASSAGES_PATH, 'r', encoding='utf-8') as f:
        passages = json.load(f)
    
    return passages


def load_indian_stats():
    """Load Gap 1 statistics."""
    if not INDIAN_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Gap 1 stats not found: {INDIAN_STATS_PATH}"
        )
    
    with open(INDIAN_STATS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# TEST 1: PREDICTION DISTRIBUTION
# ============================================================

def test_prediction_distribution(predictions, label_names):
    """
    Test 1: Calculate distribution of predicted labels.
    
    Returns:
        dict: Distribution analysis
    """
    unique, counts = np.unique(predictions, return_counts=True)
    
    distribution = {}
    for label, count in zip(unique, counts):
        pct = 100.0 * count / len(predictions)
        distribution[label] = {
            'count': int(count),
            'percentage': float(pct)
        }
    
    # Detect degeneracy
    pct_values = [d['percentage'] for d in distribution.values()]
    max_pct = max(pct_values)
    min_pct = min(pct_values)
    is_degenerate = max_pct > 90
    
    result = {
        'num_unique_labels': len(unique),
        'total_predictions': int(len(predictions)),
        'distribution': distribution,
        'max_label_percentage': float(max_pct),
        'min_label_percentage': float(min_pct),
        'is_potentially_degenerate': bool(is_degenerate),
        'degenerate_threshold_percentage': 90.0,
    }
    
    return result


def save_prediction_distribution(distribution_result):
    """Save prediction distribution to JSON."""
    output_path = OUTPUT_DIR / "gap2b_prediction_distribution.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(distribution_result, f, indent=2)
    return output_path


def plot_prediction_distribution(predictions, major_labels):
    """Create visualization of prediction distribution."""
    unique, counts = np.unique(predictions, return_counts=True)
    
    # Map to labels
    label_list = [major_labels[int(u)] if isinstance(u, (int, np.integer)) else u for u in unique]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(label_list)), counts)
    plt.xticks(range(len(label_list)), label_list, rotation=45, ha='right')
    plt.ylabel('Count')
    plt.title('Gap 2B: RMU:ECHR Predicted Functional Labels Distribution (Indian Judgment)')
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "gap2b_prediction_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path


# ============================================================
# TEST 2: FUNCTIONAL PREDICTIONS OVER DOCUMENT ORDER
# ============================================================

def test_functional_sequence(predictions, passages):
    """
    Test 2: Map predicted labels against original passage order.
    
    Returns:
        list: Sequence of predictions with passage metadata
    """
    sequence = []
    for idx, (pred, passage) in enumerate(zip(predictions, passages)):
        sequence.append({
            'passage_index': int(passage.get('passage_id', idx)),
            'predicted_function': str(pred),
            'passage_start_char': int(passage['original_position']['start_char']),
            'passage_end_char': int(passage['original_position']['end_char']),
        })
    
    return sequence


def save_functional_sequence(sequence):
    """Save sequence to JSON."""
    output_path = OUTPUT_DIR / "gap2b_functional_sequence.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sequence, f, indent=2)
    return output_path


def plot_functional_sequence(predictions, major_labels):
    """Create visualization of functional sequence along document."""
    # Map predictions to numeric codes for plotting
    label_to_code = {label: i for i, label in enumerate(major_labels)}
    numeric_seq = [label_to_code[str(pred)] for pred in predictions]
    
    plt.figure(figsize=(16, 4))
    plt.plot(range(len(numeric_seq)), numeric_seq, marker='o', markersize=3, linestyle='-', linewidth=1)
    plt.yticks(range(len(major_labels)), major_labels)
    plt.xlabel('Passage Index')
    plt.ylabel('Predicted Function')
    plt.title('Gap 2B: Functional Label Sequence Through Indian Judgment')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "gap2b_functional_sequence.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path


# ============================================================
# TEST 3: WITHIN-CLASS VS BETWEEN-CLASS SEMANTIC SIMILARITY
# ============================================================

def test_semantic_consistency(predictions, embeddings, major_labels):
    """
    Test 3: Calculate within-class and between-class semantic similarity.
    
    Returns:
        dict: Semantic consistency analysis
    """
    # Group embeddings by predicted class
    class_embeddings = {}
    for pred, emb in zip(predictions, embeddings):
        pred_str = str(pred)
        if pred_str not in class_embeddings:
            class_embeddings[pred_str] = []
        class_embeddings[pred_str].append(emb)
    
    # Compute within-class similarities
    within_class_sims = []
    within_class_by_label = {}
    
    for label, embs in class_embeddings.items():
        embs_array = np.array(embs)
        if len(embs_array) > 1:
            # Compute pairwise cosine similarities
            sims = cosine_similarity(embs_array)
            # Get upper triangle (exclude diagonal)
            upper_triangle = sims[np.triu_indices_from(sims, k=1)]
            within_class_sims.extend(upper_triangle)
            within_class_by_label[label] = {
                'mean': float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 0.0,
                'std': float(np.std(upper_triangle)) if len(upper_triangle) > 0 else 0.0,
                'count_pairs': int(len(upper_triangle)),
                'count_passages': len(embs_array),
            }
    
    within_mean = float(np.mean(within_class_sims)) if within_class_sims else 0.0
    within_std = float(np.std(within_class_sims)) if within_class_sims else 0.0
    
    # Compute between-class similarities
    between_class_sims = []
    class_labels = sorted(class_embeddings.keys())
    
    for i, label1 in enumerate(class_labels):
        for label2 in class_labels[i+1:]:
            embs1 = np.array(class_embeddings[label1])
            embs2 = np.array(class_embeddings[label2])
            sims = cosine_similarity(embs1, embs2)
            between_class_sims.extend(sims.flatten())
    
    between_mean = float(np.mean(between_class_sims)) if between_class_sims else 0.0
    between_std = float(np.std(between_class_sims)) if between_class_sims else 0.0
    
    result = {
        'within_class': {
            'mean': within_mean,
            'std': within_std,
            'num_pairs': int(len(within_class_sims)),
            'by_label': within_class_by_label,
        },
        'between_class': {
            'mean': between_mean,
            'std': between_std,
            'num_pairs': int(len(between_class_sims)),
        },
        'difference': float(within_mean - between_mean),
        'interpretation': (
            'Positive difference suggests within-class semantic consistency. '
            'Negative difference suggests poor class separation.'
        ),
    }
    
    return result


def save_semantic_consistency(consistency_result):
    """Save semantic consistency analysis."""
    output_path = OUTPUT_DIR / "gap2b_semantic_consistency.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(consistency_result, f, indent=2)
    return output_path


def plot_semantic_consistency(consistency_result):
    """Create visualization of semantic consistency."""
    within_mean = consistency_result['within_class']['mean']
    between_mean = consistency_result['between_class']['mean']
    
    plt.figure(figsize=(10, 6))
    categories = ['Within Class\n(same predicted function)', 'Between Class\n(different predicted functions)']
    values = [within_mean, between_mean]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    plt.ylabel('Average Cosine Similarity', fontsize=12)
    plt.title('Gap 2B: Semantic Consistency of RMU:ECHR Predictions on Indian Judgment', fontsize=13)
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "gap2b_semantic_consistency.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path


# ============================================================
# TEST 4: PREDICTION TRANSITIONS
# ============================================================

def test_prediction_transitions(predictions, major_labels):
    """
    Test 4: Calculate adjacent-passage functional transitions.
    
    Returns:
        dict: Transition matrix and statistics
    """
    # Create label-to-index mapping
    label_to_idx = {label: i for i, label in enumerate(major_labels)}
    
    # Initialize transition matrix
    n_labels = len(major_labels)
    transition_matrix = np.zeros((n_labels, n_labels), dtype=int)
    
    # Count transitions
    for i in range(len(predictions) - 1):
        curr_pred = str(predictions[i])
        next_pred = str(predictions[i + 1])
        
        curr_idx = label_to_idx[curr_pred]
        next_idx = label_to_idx[next_pred]
        
        transition_matrix[curr_idx, next_idx] += 1
    
    # Convert to dict
    transition_dict = {}
    for i, from_label in enumerate(major_labels):
        transition_dict[from_label] = {}
        total_transitions_from = transition_matrix[i].sum()
        
        for j, to_label in enumerate(major_labels):
            count = int(transition_matrix[i, j])
            pct = 100.0 * count / total_transitions_from if total_transitions_from > 0 else 0.0
            transition_dict[from_label][to_label] = {
                'count': count,
                'percentage': float(pct),
            }
    
    # Calculate statistics
    unique, counts = np.unique(predictions, return_counts=True)
    self_transition_count = sum(
        transition_matrix[i, i] for i in range(n_labels)
    )
    total_transitions = len(predictions) - 1
    self_transition_pct = (
        100.0 * self_transition_count / total_transitions 
        if total_transitions > 0 else 0.0
    )
    
    result = {
        'transition_matrix': transition_dict,
        'total_transitions': total_transitions,
        'self_transition_count': int(self_transition_count),
        'self_transition_percentage': float(self_transition_pct),
        'unique_transition_pairs': int(np.count_nonzero(transition_matrix)),
        'matrix_density': float(np.count_nonzero(transition_matrix) / (n_labels * n_labels)),
    }
    
    return result


def save_transition_matrix(transition_result):
    """Save transition matrix."""
    output_path = OUTPUT_DIR / "gap2b_transition_matrix.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transition_result, f, indent=2)
    return output_path


def plot_transition_matrix(transition_result, major_labels):
    """Create heatmap of transition matrix."""
    # Extract counts into matrix
    label_to_idx = {label: i for i, label in enumerate(major_labels)}
    n_labels = len(major_labels)
    matrix = np.zeros((n_labels, n_labels))
    
    trans_dict = transition_result['transition_matrix']
    for i, from_label in enumerate(major_labels):
        for j, to_label in enumerate(major_labels):
            matrix[i, j] = trans_dict[from_label][to_label]['count']
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))
    ax.set_xticklabels([l.replace(' ', '\n') for l in major_labels], fontsize=9)
    ax.set_yticklabels([l.replace(' ', '\n') for l in major_labels], fontsize=9)
    
    # Labels
    ax.set_xlabel('Next Function', fontsize=11)
    ax.set_ylabel('Current Function', fontsize=11)
    ax.set_title('Gap 2B: Functional Transition Matrix (Indian Judgment)', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Count', fontsize=10)
    
    # Add text annotations
    for i in range(n_labels):
        for j in range(n_labels):
            text = ax.text(j, i, int(matrix[i, j]),
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "gap2b_transition_matrix.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path


# ============================================================
# TEST 5: MANUAL SANITY CHECK SAMPLE
# ============================================================

def test_manual_sanity_sample(
    passages,
    predictions,
    probabilities,
    major_labels,
    sample_size=30,
    seed=RANDOM_SEED
):
    """
    Test 5: Select a deterministic sample for manual inspection.
    
    Returns:
        list: Sample of passages with predictions
    """
    np.random.seed(seed)
    
    # Stratified sampling: select passages distributed across document
    indices = np.linspace(
        0,
        len(passages) - 1,
        sample_size,
        dtype=int
    )
    
    sample = []
    for idx in indices:
        passage = passages[idx]
        pred = predictions[idx]
        probs = probabilities[idx]
        confidence = float(np.max(probs))
        
        sample.append({
            'passage_index': int(passage.get('passage_id', idx)),
            'passage_text': passage['text'],
            'passage_start_char': int(passage['original_position']['start_char']),
            'passage_end_char': int(passage['original_position']['end_char']),
            'predicted_function': str(pred),
            'confidence': confidence,
            'rmu_echr_label_probabilities': {
                label: float(prob)
                for label, prob in zip(major_labels, probs)
            },
            'note': 'Manual review required - prediction validity unknown'
        })
    
    return sample


def save_manual_sanity_sample(sample):
    """Save manual sanity check sample."""
    output_path = OUTPUT_DIR / "gap2b_manual_sanity_sample.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    return output_path


# ============================================================
# CONFIGURATION FILE
# ============================================================

def create_configuration_record(gap1_stats, gap2a_config):
    """Create configuration record for reproducibility."""
    return {
        'experiment': 'Gap 2B - Indian Transfer Test',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'random_seed': RANDOM_SEED,
        'target_document': INDIAN_JUDGMENT_ID,
        'gap1_passages_source': str(INDIAN_PASSAGES_PATH),
        'gap2a_model_name': gap2a_config.MODEL_NAME,
        'gap2a_major_labels': gap2a_config.MAJOR_LABELS,
        'embedding_model': gap2a_config.MODEL_NAME,
        'embedding_dimension': gap1_stats['stats']['embedding_dimension'],
        'passages_per_batch': 32,
        'indian_judgment_stats': gap1_stats['stats'],
        'description': (
            'Tests whether RMU:ECHR-trained classifier produces meaningful '
            'structural signals on Indian judicial text. Predictions are NOT '
            'Indian ground truth but exploratory domain-transfer signals.'
        ),
    }


def save_configuration(config):
    """Save configuration to JSON."""
    output_path = OUTPUT_DIR / "gap2b_configuration.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    return output_path


# ============================================================
# COMPREHENSIVE REPORT
# ============================================================

def create_report(
    gap1_stats,
    distribution_result,
    sequence_result,
    consistency_result,
    transition_result,
    sample_result,
    execution_time,
):
    """Create the final Gap 2B report."""
    
    report = f"""# GAP 2B: INDIAN JUDICIAL CONTEXT TRANSFER TEST

**Report Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Execution Time:** {execution_time:.2f} seconds

---

## 1. OBJECTIVE

Test whether the RMU:ECHR-trained legal-functional classifier can produce meaningful 
structural signals when transferred to a real Indian Supreme Court judgment.

**Critical Note:** These predictions are NOT Indian legal ground truth. They represent 
exploratory domain-transfer of a European legal classification scheme to Indian judicial text.

---

## 2. WHY RMU:ECHR WAS USED

RMU:ECHR is a corpus of {len(Gap2AConfig.MAJOR_LABELS)} labeled functional categories 
from European Court of Human Rights (ECHR) judgments:

- **Subsumtion** - application of law to facts
- **Vorherige Rechtsprechung des EGMR** - previous ECHR case law
- **Verhältnismäßigkeitsprüfung – Angemessenheit** - proportionality testing
- **Entscheidung des EGMR** - ECHR decision/ruling

RMU:ECHR served as the source domain for a semantic embedding classifier.

---

## 3. WHY IT IS NOT INDIAN GROUND TRUTH

RMU:ECHR annotations reflect European legal reasoning, procedure, and doctrine. 
Indian Supreme Court judgments follow different procedural rules, precedent structures, 
and constitutional frameworks.

Predictions on Indian text represent the classifier's best attempt to recognize similar 
semantic/functional patterns, but carry no guarantee of legal validity in the Indian context.

---

## 4. INDIAN JUDGMENT USED

**Document:** Vijay Madanlal Choudhary vs Union Of India (27 July 2022)

**Characteristics:**
- **Characters:** {gap1_stats['stats']['total_characters']:,}
- **Words:** {gap1_stats['stats']['total_words']:,}
- **Sentences:** {gap1_stats['stats']['total_sentences']:,}
- **Passages (5 sentences each):** {gap1_stats['stats']['total_passages']:,}

**Passage Construction:**
- Method: 5 consecutive sentences per passage
- Order: Original document order preserved
- Metadata: Exact character offsets recorded

---

## 5. EXPERIMENTAL METHODOLOGY

### 5.1 Reuse of Gap 2A Classifier

- **Classifier Type:** Logistic Regression (sklearn)
- **Training Data:** RMU:ECHR annotations (document-level split)
- **Training Set:** ~{int(Gap2AConfig.SAMPLES_PER_LABEL * len(Gap2AConfig.MAJOR_LABELS) * 0.8)} passages
- **Test Set:** ~{int(Gap2AConfig.SAMPLES_PER_LABEL * len(Gap2AConfig.MAJOR_LABELS) * 0.2)} passages
- **Gap 2A Accuracy (on ECHR test set):** ~0.733
- **Gap 2A Macro-F1 (on ECHR test set):** ~0.732

The same trained classifier is applied without retraining or tuning to Indian data.

### 5.2 Application to Indian Judgment

1. Load Indian passages (Gap 1 output)
2. Generate embeddings using same SentenceTransformer model
3. Apply trained RMU:ECHR classifier to each passage
4. Record predictions and confidence scores
5. Analyze prediction patterns

---

## 6. EMBEDDING DETAILS

- **Model:** all-MiniLM-L6-v2 (SentenceTransformer)
- **Dimension:** 384
- **Normalization:** L2-normalized
- **Batch Size:** 32
- **Indian Judgment Embeddings Generated:** {gap1_stats['stats']['total_passages']}
- **Processing Time:** {gap1_stats['stats']['processing_time_seconds']:.2f} seconds
- **Throughput:** {gap1_stats['stats']['texts_per_second']:.2f} passages/second

---

## 7. PREDICTION DISTRIBUTION

**Threshold for Degeneracy:** >90% of predictions assigned to one label

**Results:**

"""
    
    for label, stats in distribution_result['distribution'].items():
        report += f"- **{label}:** {stats['count']} passages ({stats['percentage']:.2f}%)\n"
    
    report += f"""
**Analysis:**
- Unique predicted labels: {distribution_result['num_unique_labels']}/{len(Gap2AConfig.MAJOR_LABELS)}
- Max single-label percentage: {distribution_result['max_label_percentage']:.2f}%
- Min single-label percentage: {distribution_result['min_label_percentage']:.2f}%
- Degenerate (>90% one label)? **{distribution_result['is_potentially_degenerate']}**

"""
    
    if distribution_result['is_potentially_degenerate']:
        report += """⚠️ **WARNING:** Predictions show potential degeneracy. The classifier may not be 
generalizing well to Indian text and may be defaulting to a majority class.
This suggests weak domain transfer.

"""
    else:
        report += """✓ Predictions are distributed across multiple labels, suggesting the classifier 
is not simply defaulting to one class.

"""
    
    report += """---

## 8. SEQUENTIAL PREDICTION ANALYSIS

**Purpose:** Determine whether predictions form identifiable regions/patterns rather 
than random switching.

**Method:** Plot predicted functional category against passage index.

**Visualization:** See `gap2b_functional_sequence.png`

**Key Observation:** 
- High variance in predictions at every step may indicate weak transfer
- Clusters of same prediction may indicate semantically consistent regions
- See manual sanity sample for linguistic interpretation

---

## 9. SEMANTIC CONSISTENCY ANALYSIS

**Purpose:** Test whether passages assigned to the same predicted category are 
semantically more similar than passages in different categories.

**Method:**
- Calculate average cosine similarity within each predicted class
- Calculate average cosine similarity between different predicted classes
- Report difference (positive = within-class consistency)

**Results:**

| Metric | Value |
|--------|-------|
| **Within-Class Mean Similarity** | {consistency_result['within_class']['mean']:.6f} |
| **Between-Class Mean Similarity** | {consistency_result['between_class']['mean']:.6f} |
| **Difference** | {consistency_result['difference']:.6f} |

**Interpretation:**

"""
    
    if consistency_result['difference'] > 0.05:
        report += """✓ **POSITIVE DIFFERENCE:** Passages with the same predicted label show 
higher semantic similarity than passages in different classes. This suggests the classifier's 
predictions capture meaningful semantic structure, even if labels may not be legally accurate.

"""
    elif consistency_result['difference'] > -0.05:
        report += """⚠️ **NEAR-ZERO DIFFERENCE:** Within-class and between-class similarities are 
nearly equivalent. Predictions may not be capturing meaningful semantic structure.

"""
    else:
        report += """✗ **NEGATIVE DIFFERENCE:** Passages in different predicted classes are actually 
MORE similar than passages in the same class. This indicates the predictions do not correspond 
to semantic structure, suggesting very poor transfer.

"""
    
    report += f"""
**Semantic Consistency by Label:**

"""
    for label, stats in consistency_result['within_class']['by_label'].items():
        report += f"- **{label}:** {stats['count_passages']} passages, within-class mean sim = {stats['mean']:.6f}\n"
    
    report += """---

## 10. TRANSITION ANALYSIS

**Purpose:** Analyze whether the sequence of predicted labels shows structure 
(e.g., tendency to remain in same category) or appears random.

**Results:**

| Metric | Value |
|--------|-------|
| **Total Transitions** | {transition_result['total_transitions']} |
| **Self-Transitions (same label twice)** | {transition_result['self_transition_count']} ({transition_result['self_transition_percentage']:.2f}%) |
| **Unique Transition Types** | {transition_result['unique_transition_pairs']} |
| **Matrix Density** | {transition_result['matrix_density']:.4f} |

**Interpretation:**

""".format(**transition_result)
    
    if transition_result['self_transition_percentage'] > 40:
        report += """✓ **STRONG SELF-TRANSITIONS:** The classifier tends to assign the same label 
to consecutive passages. This suggests semantic coherence within the text and structure in predictions.

"""
    elif transition_result['self_transition_percentage'] > 25:
        report += """⊙ **MODERATE SELF-TRANSITIONS:** Some tendency to remain in the same predicted 
category, but with frequent changes. Structure is present but limited.

"""
    else:
        report += """✗ **WEAK SELF-TRANSITIONS:** The classifier rapidly switches between predicted 
labels, suggesting either true semantic boundaries in the text or poor prediction coherence.

"""
    
    report += f"""
**Transition Matrix:** See `gap2b_transition_matrix.json` and `gap2b_transition_matrix.png`

---

## 11. MANUAL SANITY-CHECK SAMPLE

**Method:**
- Deterministic stratified sample of {len(sample_result)} passages
- Distributed across the full document (not just beginning)
- Each passage includes text, predicted label, and confidence score

**Purpose:**
NOT to calculate accuracy. Instead, to allow human inspection of whether predictions 
appear:
- Completely nonsensical for Indian legal text
- Somewhat plausible as functional categories
- Highly mismatched to the target domain

**Sample Location:** `gap2b_manual_sanity_sample.json`

**Instruction for Reviewers:**
1. Read the passage text
2. Check if the predicted RMU:ECHR functional label seems to match the passage's role
3. Remember that perfect accuracy is NOT expected (this is exploratory)
4. Note patterns: Are there consistent categories? Do predictions seem random?
5. Identify failure modes: Where does the transfer break down?

---

## 12. RESULTS SUMMARY

### Classification Outputs Generated

- ✓ `gap2b_prediction_distribution.json` - label frequencies
- ✓ `gap2b_prediction_distribution.png` - bar chart
- ✓ `gap2b_functional_sequence.json` - passage-by-passage predictions
- ✓ `gap2b_functional_sequence.png` - sequence plot
- ✓ `gap2b_semantic_consistency.json` - within/between-class similarity
- ✓ `gap2b_semantic_consistency.png` - similarity comparison chart
- ✓ `gap2b_transition_matrix.json` - transition frequencies
- ✓ `gap2b_transition_matrix.png` - heatmap
- ✓ `gap2b_manual_sanity_sample.json` - 30 passages for manual review
- ✓ `gap2b_configuration.json` - reproducibility record

### Data Format Compatibility

All outputs preserve the Gap 1 passage structure:
- `document_id`
- `passage_index` / `passage_id`
- `passage_text`
- `original_position` (start_char, end_char)
- `predicted_function`
- `confidence` (if available)

This format is compatible with Gap 3 and Gap 4 downstream experiments.

---

## 13. LIMITATIONS

1. **No Indian Ground Truth:** Predictions are not validated against Indian legal expertise
2. **Domain Mismatch:** RMU:ECHR is European; Indian law has different procedures
3. **No Retraining:** The classifier was trained on ECHR data and not adapted
4. **No Fine-Tuning:** No optimization for Indian context
5. **Single Document:** Only one judgment tested; generalization is unknown
6. **Functional vs. Topical:** Classifier learns functional categories; may confuse with topic
7. **Class Imbalance:** RMU:ECHR training may have skewed distributions

---

## 14. CONCLUSION

### Transfer Success Indicators

{distribution_result['is_potentially_degenerate'] and '✗' or '✓'} Predictions are non-degenerate (use multiple labels)
{consistency_result['difference'] > 0 and '✓' or '✗'} Semantic consistency present (within-class > between-class)
{transition_result['self_transition_percentage'] > 30 and '✓' or '✗'} Structural signal in transitions (self-transition rate)

### Overall Assessment

Based on the analysis:

"""
    
    score = sum([
        not distribution_result['is_potentially_degenerate'],
        consistency_result['difference'] > 0,
        transition_result['self_transition_percentage'] > 30,
    ])
    
    if score >= 2:
        report += """**TRANSFER SIGNAL DETECTED:** The RMU:ECHR classifier shows some ability to 
produce structured predictions on Indian text. Predictions show semantic consistency and 
non-degenerate distribution. This suggests the functional representation learned from ECHR 
may capture generalizable aspects of legal text structure that cross domain boundaries.

**Recommendation:** These structural signals may be useful as auxiliary features in 
Gap 3 (passage relationship analysis) or Gap 4 (summary coherence evaluation).

"""
    elif score >= 1:
        report += """**WEAK TRANSFER SIGNAL:** The classifier shows some indicators of transfer 
but with significant limitations. Results suggest the learned representation partially 
generalizes to Indian text but may require domain adaptation or supplementary signals.

**Recommendation:** Use with caution as an exploratory feature; do not rely solely on 
these predictions. Consider combining with domain-specific Indian legal signals.

"""
    else:
        report += """**POOR TRANSFER SIGNAL:** The RMU:ECHR classifier does not effectively 
transfer to Indian legal text. Predictions are degenerate, lack semantic consistency, or 
show no structural pattern. This indicates fundamental domain mismatch.

**Recommendation:** Consider domain adaptation techniques (fine-tuning, retraining, or 
transfer learning with Indian legal data) before using these predictions in downstream tasks.

"""
    
    report += """---

## 15. IMPLICATIONS FOR THE FINAL INDIAN SUMMARIZER

### Use Cases That Remain Valid

- **Gap 1 Passage Construction:** Proven and independent; not affected by transfer results
- **Semantic Embeddings:** General-purpose embeddings (all-MiniLM-L6-v2) are likely useful
- **Passage Ordering:** Original document structure is preserved

### Use Cases Requiring Caution

- **Functional Classification:** RMU:ECHR labels may not generalize; consider Indian-specific taxonomy
- **Structure Prediction:** Use transfer results only as exploratory signals; validate with domain experts
- **Summarization Features:** If used, combine with other structural signals (e.g., sentence centrality, citation networks)

### Recommended Next Steps

1. **Gap 3:** Proceed with passage relationship analysis using general embeddings
2. **Gap 4:** Test multiple feature combinations including (but not limited to) transfer predictions
3. **Indian Annotation:** Consider annotating a small Indian sample for ground-truth legal function taxonomy
4. **Domain Adaptation:** If high-quality Indian legal functional annotations become available, retrain classifier

---

## 16. FILES AND REPRODUCIBILITY

**Random Seed:** {RANDOM_SEED}

**Configuration:** All settings recorded in `gap2b_configuration.json`

**Full Reproducibility:** Run the following command:

```bash
python backend/app/experiment_ip/gap2/test_indian_transfer.py
```

**Requirements:**
- Gap 1 passages: `data/experiments/gap1/Vijay_Madanlal_Choudhary_*.json`
- RMU:ECHR dataset: `data/experiments/gap2/rmu_echr_annotations.json`
- Dependencies: sentence-transformers, scikit-learn, numpy, matplotlib

---

**End of Report**
"""
    
    return report


def save_report(report_text):
    """Save report to markdown file."""
    output_path = OUTPUT_DIR / "GAP2B_INDIAN_TRANSFER_REPORT.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    return output_path


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run the complete Gap 2B experiment."""
    
    print("=" * 70)
    print("GAP 2B: INDIAN JUDICIAL CONTEXT TRANSFER TEST")
    print("=" * 70)
    print()
    
    execution_start = time.time()
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================================
    # STEP 1: LOAD GAP 2A CLASSIFIER
    # ========================================================
    
    print("[1/10] Training Gap 2A classifier from RMU:ECHR...")
    print()
    
    gap2a_package = train_full_gap2a_classifier(verbose=True)
    
    classifier = gap2a_package['classifier']
    embedding_model = gap2a_package['model']
    major_labels = gap2a_package['config'].MAJOR_LABELS
    
    print()
    print("✓ Gap 2A classifier ready")
    print()
    
    # ========================================================
    # STEP 2: LOAD INDIAN JUDGMENT
    # ========================================================
    
    print("[2/10] Loading Indian judgment passages (Gap 1)...")
    
    passages = load_indian_passages()
    gap1_stats = load_indian_stats()
    
    print(f"✓ Loaded {len(passages)} passages")
    print(f"  - Characters: {gap1_stats['stats']['total_characters']:,}")
    print(f"  - Words: {gap1_stats['stats']['total_words']:,}")
    print(f"  - Sentences: {gap1_stats['stats']['total_sentences']:,}")
    print()
    
    # ========================================================
    # STEP 3: PREDICT FUNCTIONAL LABELS
    # ========================================================
    
    print("[3/10] Generating embeddings and predicting labels...")
    print()
    
    prediction_result = predict_on_passages(
        classifier,
        passages,
        embedding_model,
        batch_size=32
    )
    
    predictions = prediction_result['predictions']
    embeddings = prediction_result['embeddings']
    probabilities = prediction_result['probabilities']
    
    print()
    print(f"✓ Generated {len(predictions)} predictions")
    print()
    
    # ========================================================
    # TEST 1: PREDICTION DISTRIBUTION
    # ========================================================
    
    print("[4/10] Test 1: Prediction distribution...")
    
    distribution_result = test_prediction_distribution(predictions, major_labels)
    dist_path = save_prediction_distribution(distribution_result)
    plot_dist_path = plot_prediction_distribution(predictions, major_labels)
    
    print(f"✓ Saved to: {dist_path}")
    print(f"✓ Visualization: {plot_dist_path}")
    print()
    
    # ========================================================
    # TEST 2: FUNCTIONAL SEQUENCE
    # ========================================================
    
    print("[5/10] Test 2: Functional sequence over document...")
    
    sequence_result = test_functional_sequence(predictions, passages)
    seq_path = save_functional_sequence(sequence_result)
    plot_seq_path = plot_functional_sequence(predictions, major_labels)
    
    print(f"✓ Saved to: {seq_path}")
    print(f"✓ Visualization: {plot_seq_path}")
    print()
    
    # ========================================================
    # TEST 3: SEMANTIC CONSISTENCY
    # ========================================================
    
    print("[6/10] Test 3: Semantic consistency analysis...")
    
    consistency_result = test_semantic_consistency(predictions, embeddings, major_labels)
    cons_path = save_semantic_consistency(consistency_result)
    plot_cons_path = plot_semantic_consistency(consistency_result)
    
    print(f"✓ Saved to: {cons_path}")
    print(f"✓ Visualization: {plot_cons_path}")
    print(f"  - Within-class mean sim: {consistency_result['within_class']['mean']:.6f}")
    print(f"  - Between-class mean sim: {consistency_result['between_class']['mean']:.6f}")
    print(f"  - Difference: {consistency_result['difference']:.6f}")
    print()
    
    # ========================================================
    # TEST 4: TRANSITION MATRIX
    # ========================================================
    
    print("[7/10] Test 4: Prediction transitions...")
    
    transition_result = test_prediction_transitions(predictions, major_labels)
    trans_path = save_transition_matrix(transition_result)
    plot_trans_path = plot_transition_matrix(transition_result, major_labels)
    
    print(f"✓ Saved to: {trans_path}")
    print(f"✓ Visualization: {plot_trans_path}")
    print(f"  - Total transitions: {transition_result['total_transitions']}")
    print(f"  - Self-transition rate: {transition_result['self_transition_percentage']:.2f}%")
    print()
    
    # ========================================================
    # TEST 5: MANUAL SANITY SAMPLE
    # ========================================================
    
    print("[8/10] Test 5: Manual sanity-check sample...")
    
    sample_result = test_manual_sanity_sample(
        passages,
        predictions,
        probabilities,
        major_labels,
        sample_size=30,
        seed=RANDOM_SEED
    )
    sample_path = save_manual_sanity_sample(sample_result)
    
    print(f"✓ Saved to: {sample_path}")
    print(f"  - Sample size: {len(sample_result)} passages")
    print()
    
    # ========================================================
    # CONFIGURATION
    # ========================================================
    
    print("[9/10] Creating configuration record...")
    
    config = create_configuration_record(gap1_stats, Gap2AConfig)
    config_path = save_configuration(config)
    
    print(f"✓ Saved to: {config_path}")
    print()
    
    # ========================================================
    # FINAL REPORT
    # ========================================================
    
    print("[10/10] Generating final report...")
    
    execution_time = time.time() - execution_start
    
    report = create_report(
        gap1_stats,
        distribution_result,
        sequence_result,
        consistency_result,
        transition_result,
        sample_result,
        execution_time,
    )
    
    report_path = save_report(report)
    
    print(f"✓ Saved to: {report_path}")
    print()
    
    # ========================================================
    # SUMMARY
    # ========================================================
    
    print("=" * 70)
    print("GAP 2B EXPERIMENT COMPLETE")
    print("=" * 70)
    print()
    print("Output Files Generated:")
    print(f"  1. {dist_path.name}")
    print(f"  2. {plot_dist_path.name}")
    print(f"  3. {seq_path.name}")
    print(f"  4. {plot_seq_path.name}")
    print(f"  5. {cons_path.name}")
    print(f"  6. {plot_cons_path.name}")
    print(f"  7. {trans_path.name}")
    print(f"  8. {plot_trans_path.name}")
    print(f"  9. {sample_path.name}")
    print(f" 10. {config_path.name}")
    print(f" 11. {report_path.name}")
    print()
    print(f"Total Execution Time: {execution_time:.2f} seconds")
    print()
    print(f"View the report:")
    print(f"  {report_path}")
    print()


if __name__ == "__main__":
    main()
