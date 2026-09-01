# Gap 3: Passage Relationship Signal Experiment

## Overview

Gap 3 investigates whether relationships between legally functional passages can be detected using combinations of:

1. **Semantic similarity** — cosine similarity of passage embeddings
2. **Document proximity** — passage index distance and character distance
3. **Legal-function information** — from Gap 2 (actor and argument-type labels)
4. **Actor information** — from RMU:ECHR annotations

## Purpose

This is an **exploratory experiment** to determine whether these signals provide evidence that a passage-level reasoning graph is feasible. We are NOT building a final relationship classifier, and we are NOT claiming that RMU:ECHR provides gold relationship labels.

## Key Constraints

- Do NOT modify Gap 1 or Gap 2
- Do NOT invent relationship labels
- Do NOT treat actor/type transitions as causal relationships
- Clearly distinguish OBSERVED STRUCTURAL PATTERNS from TRUE LEGAL RELATIONSHIPS
- RMU:ECHR is used for signal extraction, NOT as ground truth

## Data Sources

- **Gap 2 Annotations**: `data/experiments/gap2/rmu_echr_annotations.json`
  - 15,211 legal annotations
  - 375 documents
  - 5 actors
  - 15 argument types

## Experiment Phases

### Phase 1: Passage Construction
- Load RMU:ECHR annotations
- Group by document_id
- Build passage-level representations from existing infrastructure
- Attach overlapping annotations to passages

### Phase 2: Pair Candidate Definition
- Construct local candidate pairs (nearby passages within same document)
- Construct distant baseline pairs (far passages in same document)
- Do NOT make cross-document comparisons initially

### Phase 3: Feature Computation
For each pair, compute:
- Semantic cosine similarity
- Passage distance (index difference)
- Character distance
- Actor transitions
- Argument-type transitions

### Phase 4: Analysis
- Semantic similarity by distance
- Actor transition matrix
- Argument-type transition analysis
- Combined signal analysis
- Local vs distant comparison

### Phase 5: Visualization & Reporting
- Similarity vs distance plots
- Transition matrices
- Structured JSON results
- Terminal summary report

## Output

Results saved to `data/experiments/gap3/`:
- `gap3_relationship_analysis.json` — comprehensive numerical results
- Visualization PNG files

## Running the Experiment

```bash
cd backend/app/experiment_ip/gap3
python test_passage_relationships.py
```

## Important Notes

- Results are DESCRIPTIVE only
- No causal claims are made
- Effect sizes and confidence intervals are provided where appropriate
- All limitations are explicitly stated
- Experiment is reproducible with fixed random seed
