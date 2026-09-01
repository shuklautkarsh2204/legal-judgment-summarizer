# Gap 3 Experiment — Quick Start Guide

## Overview

Gap 3 is an exploratory experiment to investigate whether relationships between legal passages can be detected using:

- **Semantic similarity** (embedding-based)
- **Document proximity** (passage distance)
- **Actor information** (from RMU:ECHR)
- **Argument-type information** (from RMU:ECHR)

## Important Constraints

✓ Do NOT modify Gap 1 or Gap 2  
✓ Do NOT invent relationship labels  
✓ Do NOT treat transitions as causal relationships  
✓ Results are DESCRIPTIVE ONLY  

## Project Structure

```
backend/app/experiment_ip/gap3/
├── __init__.py                          # Package marker
├── README.md                            # Overview (this file)
├── QUICKSTART.md                        # This file
├── test_passage_relationships.py        # Main experiment script
└── diagnose.py                          # Diagnostic utility

data/experiments/gap3/                   # Output directory (created when run)
├── gap3_relationship_analysis.json      # Results
└── [visualization files, if added]
```

## Running the Experiment

### Prerequisites

Ensure dependencies are installed:

```bash
pip install sentence-transformers scikit-learn numpy
```

### Standard Run

From the project root:

```bash
cd d:\Projects\LegalJudgementSummarizer
python backend/app/experiment_ip/gap3/test_passage_relationships.py
```

### Diagnostic Run

To test if imports and environment are set up correctly:

```bash
python backend/app/experiment_ip/gap3/diagnose.py
```

## Experiment Pipeline

The experiment proceeds through these phases:

### 1. Load Annotations
- Reads `data/experiments/gap2/rmu_echr_annotations.json`
- Groups by document_id
- Reports statistics (15,211 annotations, 375 documents, 5 actors, 15 argument types)

### 2. Build Passages
- Constructs synthetic documents from annotation text snippets
- Groups annotations by document_id
- Sorts by character position to maintain order
- Builds passages (5 sentences per passage by default)
- Encodes with sentence-transformers (`all-MiniLM-L6-v2`)

### 3. Construct Candidate Pairs
- **Local pairs**: passages within a window (default: 10 passages apart)
- **Distant pairs**: passages far apart (sampled at ~15% rate)
- Pairs are within the same document only

### 4. Compute Pair Features
For each pair:
- Semantic cosine similarity (0-1)
- Passage distance (passage index difference)
- Character distance
- Actor transitions (actor_i → actor_j)
- Argument-type transitions (type_i → type_j)

### 5. Analyze Results
- Semantic similarity grouped by distance
- Local vs distant comparison (effect size calculation)
- Actor transition matrix
- Argument-type transition matrix
- Combined affinity score (exploratory)

### 6. Generate Output
- **gap3_relationship_analysis.json** — comprehensive results
- Terminal report with key findings

## Understanding the Results

### Semantic Similarity by Distance
Shows whether closer passages tend to have higher semantic similarity.

Example output:
```
distance_1:
  Pairs: 2348
  Mean: 0.5234
  Median: 0.5089
  Std: 0.1245

distance_2-3:
  Pairs: 1823
  Mean: 0.4912
  ...

distant_baseline:
  Pairs: 412
  Mean: 0.3845
  ...
```

**Interpretation**: If local pairs have significantly higher similarity than distant pairs, this suggests semantic structure correlates with document position.

### Local vs Distant Comparison

Reports:
- Mean similarity difference
- Effect size (Cohen's d)
- Interpretation (negligible / small / medium / large)

**Example**: 
```
Mean difference: 0.0823
Effect size: 0.6234 (medium)
```

This would suggest a **medium effect** — local passages are substantially more similar than distant passages.

### Actor and Type Transitions

Shows most frequent transitions:
```
Beschwerdeführer → EGMR: 1823 (18.3%)
EGMR → Staat: 1245 (12.5%)
Staat → EGMR: 1089 (10.9%)
...
```

**Important**: These are OBSERVED PATTERNS, not causal relationships.

### Combined Affinity Score

Combines multiple signals:
- Semantic similarity
- Inverse passage distance
- Actor compatibility
- Argument-type compatibility

Range: 0-1 (higher = more "affinity" between passages)

**Note**: This is exploratory. No machine learning is involved.

## Output Files

### gap3_relationship_analysis.json

Structure:
```json
{
  "metadata": {
    "experiment": "Gap 3 - Passage Relationship Signals",
    "timestamp": "2026-01-15 14:32:10",
    "random_seed": 42
  },
  "dataset_statistics": {...},
  "pair_statistics": {...},
  "semantic_similarity_analysis": {...},
  "local_vs_distant": {...},
  "actor_transitions": {...},
  "argument_type_transitions": {...},
  "combined_signal_analysis": {...},
  "limitations": [...]
}
```

All numerical results are float or int, suitable for downstream analysis.

## Key Findings Format

Terminal output will include:

```
======================================================================
RMU:ECHR GAP 3 — PASSAGE RELATIONSHIP SIGNAL EXPERIMENT
======================================================================

Dataset statistics
...

Pair Construction
...

Semantic Similarity by Distance
...

Local vs Distant Comparison
...

Actor Transitions
...

Argument Type Transitions
...

Combined Structural Signal
...

Experiment Summary
...
```

## Interpreting Results

### Positive Evidence for Relationship Graph
- Local pairs have **significantly higher** semantic similarity than distant pairs
- Actor/type transitions show **clear patterns**
- Combined affinity score is **high for local pairs, low for distant**
- Effect size (Cohen's d) is **≥0.5** (small effect or larger)

### Weak Evidence
- Local vs distant similarity difference is **small** (Cohen's d < 0.2)
- Transitions are **random or uniform**
- No clear pattern in affinity scores

### Negative Evidence
- No difference or **reverse pattern** (distant pairs more similar)
- Results suggest semantic similarity alone doesn't indicate relationships

## Important Limitations

1. **No gold labels**: RMU:ECHR is NOT a ground-truth passage relationship database
2. **Synthetic documents**: Passages are constructed from annotation snippets, not actual judgment texts
3. **Single perspective**: Only looking at RMU:ECHR annotations; other relationship signals may exist
4. **No causal inference**: Observed patterns don't imply causation
5. **Limited document coverage**: Analysis uses only documents with RMU:ECHR annotations

## Next Steps (If Proceeding Beyond Experiment)

If results suggest feasibility:

1. Collect actual full judgment texts for ECHR cases
2. Evaluate impact of different passage boundary strategies
3. Investigate whether Gap 2 legal-function predictions improve signal
4. Consider alternative similarity metrics (e.g., legal-domain embeddings)
5. Design passage-level relationship graph structure
6. Implement relationship classifier with appropriate validation

If results are negative:

1. Investigate better contextual representations
2. Consider document-level (rather than passage-level) analysis
3. Explore alternative definition of "relationship"

## Configuration

Parameters in `test_passage_relationships.py`:

```python
# Passage grouping
PassageBuilder(sentences_per_passage=5)

# Local window for candidate pairs
construct_candidate_pairs(local_window=10, distant_sampling_rate=0.15)

# Embedding model
SemanticAnalyzer().model = "sentence-transformers/all-MiniLM-L6-v2"

# Batch size for encoding
encode_passages(batch_size=32)

# Random seed (for reproducibility)
RANDOM_SEED = 42
```

To modify these, edit the values in the script before running.

## Troubleshooting

### "ModuleNotFoundError: No module named 'backend'"

Ensure you're running from the project root directory:
```bash
cd d:\Projects\LegalJudgementSummarizer
```

### "FileNotFoundError: data/experiments/gap2/rmu_echr_annotations.json"

Verify the file exists:
```bash
Test-Path data/experiments/gap2/rmu_echr_annotations.json
```

### Script hangs or runs slowly

- SemanticAnalyzer downloads models on first run (~50MB)
- First execution may take several minutes
- Subsequent runs are faster (model cached)
- Check system RAM/disk space

### No output after starting script

- Check if script is still running (CPU usage)
- Model download can take time on first run
- Try running `diagnose.py` first

## Contact / Questions

For questions about this experiment:
1. Check README.md in this directory
2. Review inline comments in `test_passage_relationships.py`
3. Examine JSON output format and metadata
