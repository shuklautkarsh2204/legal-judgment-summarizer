# Gap 3 Technical Specification

## Experiment Architecture

### Data Flow

```
RMU:ECHR Annotations (JSON)
    ↓
Load Annotations → Group by doc_id
    ↓
Construct Synthetic Documents (annotation text concatenation)
    ↓
Build Passages (5 sentences/passage, preserve position)
    ↓
Encode Passages (sentence-transformers all-MiniLM-L6-v2, 384-dim)
    ↓
Construct Candidate Pairs (local + distant within same document)
    ↓
Compute Features for all pairs
    ↓
Run Analyses (semantic, transitions, combined signal)
    ↓
Generate Results JSON + Terminal Report
```

## Core Classes

### Annotation

```python
@dataclass
class Annotation:
    document_id: str       # ECHR case ID (e.g., "001-100469")
    text: str              # Annotation text snippet
    begin: int             # Character offset in original judgment
    end: int               # Character offset in original judgment
    actor: str             # One of 5 actors (Beschwerdeführer, EGMR, Staat, Dritte, etc.)
    argument_type: str     # One of 15 argument types (Subsumtion, Verhältnismäßigkeitsprüfung, etc.)
```

### PassageInfo

```python
@dataclass
class PassageInfo:
    document_id: str
    passage_index: int                         # Sequential passage ID within document
    text: str                                  # Full passage text
    begin_char: int                            # Start position in synthetic document
    end_char: int                              # End position in synthetic document
    embedding: np.ndarray                      # 384-dimensional embedding
    overlapping_annotations: List[Annotation]  # Annotations that overlap this passage
```

Methods:
- `get_actors()` → List[str]  — unique actors in overlapping annotations
- `get_argument_types()` → List[str]  — unique argument types
- `primary_actor()` → Optional[str]  — single actor if exactly one
- `primary_argument_type()` → Optional[str]  — single type if exactly one

### PassagePair

```python
@dataclass
class PassagePair:
    doc_id: str
    passage_i: PassageInfo
    passage_j: PassageInfo
    pair_type: str                     # 'local' or 'distant'
    
    # Computed features
    semantic_similarity: float         # cosine similarity, range [0, 1]
    passage_distance: int              # |j - i| (passage indices)
    char_distance: int                 # character offset distance
    same_actor: Optional[bool]         # True, False, or None (Unknown)
    actor_transition: Optional[Tuple]  # (actor_i, actor_j) or None
    same_argument_type: Optional[bool] # True, False, or None
    argument_type_transition: Optional[Tuple]  # (type_i, type_j) or None
```

### Gap3Analyzer

Main orchestrator class with methods:

| Method | Purpose |
|--------|---------|
| `load_annotations()` | Load RMU:ECHR JSON, compute statistics |
| `build_passages_for_docs()` | Create passages, encode with embeddings, map annotations |
| `construct_candidate_pairs()` | Build local + distant pair candidates |
| `compute_pair_features()` | Compute all features for each pair |
| `analyze_semantic_similarity()` | Group similarity by distance, compute stats |
| `compare_local_vs_distant()` | Effect size calculation (Cohen's d) |
| `analyze_actor_transitions()` | Transition matrix and frequencies |
| `analyze_argument_type_transitions()` | Transition matrix and frequencies |
| `combined_signal_analysis()` | Exploratory affinity score |
| `generate_results_json()` | Save comprehensive results |
| `print_summary_report()` | Terminal output |

## Passage Construction Algorithm

### Input
Synthetic document text (concatenated annotation snippets)

### Process
1. **Sentence Splitting**: `split_into_sentences()` → List[str]
2. **Sentence Span Mapping**: Map each sentence to (start_char, end_char) in document
3. **Grouping**: Group consecutive sentences into passages (N per passage)
4. **Annotation Mapping**: For each passage, find overlapping annotations

Key property: Preserves original character positions for annotation alignment

### Output
List[PassageInfo] with embeddings

## Pair Candidate Selection

### Local Pairs
For each document, for each passage i:
- Consider passages j where: j > i AND j < i + local_window
- Creates pairs (i, j) marked as 'local'

Example (local_window=10):
```
Passage 0 paired with: 1, 2, 3, ..., 9 (if exist)
Passage 1 paired with: 2, 3, 4, ..., 10
Passage 5 paired with: 6, 7, 8, ..., 14
```

### Distant Pairs
All pairs (i, j) where j >= i + local_window

**Sampling**: Take ~15% random sample to balance computational cost

### Rationale
- **Local**: Potentially related (nearby in document)
- **Distant**: Baseline for comparison (far apart)

## Feature Computation

### Semantic Similarity
```
sim = cos(embedding_i, embedding_j)
    = dot(e_i, e_j) / (||e_i|| * ||e_j||)
Range: [-1, 1], typically [0, 1] for text
```

Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)

### Passage Distance
```
dist = |passage_index_j - passage_index_i|
```

### Character Distance
```
char_dist = max(passage_j.begin - passage_i.end, 0)
          | or passage_i.begin - passage_j.end if i > j
```

### Actor Information
```
actors_i = get_actors(passage_i)
actors_j = get_actors(passage_j)

same_actor = (actors_i ∩ actors_j) ≠ ∅    if both have actors
           = None                          if either is empty

actor_transition = (actors_i[0], actors_j[0])  if both have actors
```

### Argument Type Information
Similar to actor information

## Analysis Methods

### Semantic Similarity by Distance

**Grouping**:
```
distance_1  → pairs where passage_distance == 1
distance_2  → pairs where passage_distance == 2
...
distance_N  → pairs where passage_distance == N
distant_baseline → pairs where passage_distance >= local_window
```

**Metrics per group**:
- count, mean, median, std
- percentiles: p25, p75

### Local vs Distant Comparison

**Statistics**:
```
local_mean = mean(similarities for pair_type='local')
distant_mean = mean(similarities for pair_type='distant')
difference = local_mean - distant_mean

pooled_std = sqrt((std_local² + std_distant²) / 2)
Cohen's d = (local_mean - distant_mean) / pooled_std
```

**Interpretation**:
- |d| < 0.2 → negligible
- 0.2 ≤ |d| < 0.5 → small
- 0.5 ≤ |d| < 0.8 → medium
- |d| ≥ 0.8 → large

### Transition Analysis

**Actor transitions**:
```
for each pair where both have actors:
    transitions[(actor_i, actor_j)] += 1

output: sorted by frequency
```

**Result**: Transition matrix (not computed as matrix, but as frequency table)

### Combined Affinity Score

**Components** (each normalized to similar scale):
1. `sim_score = semantic_similarity` (range ≈ [0, 1])
2. `dist_score = 1 / (1 + passage_distance)` (range [0, 1])
3. `actor_score = 1 if same_actor else -0.5` (binary when available)
4. `type_score = 1 if same_type else -0.5` (binary when available)

**Formula**:
```
combined = mean([sim_score, dist_score, actor_score*, type_score*])
           * include components that are available (non-zero/defined)
```

**Rationale**: Explores whether combining signals produces better separation between local and distant pairs

## Output Format

### JSON Structure
```json
{
  "metadata": {
    "experiment": "Gap 3 - Passage Relationship Signals",
    "timestamp": "ISO 8601",
    "random_seed": 42
  },
  "dataset_statistics": {
    "total_annotations": 15211,
    "unique_documents": 375,
    "total_passages": <int>,
    "unique_actors": 5,
    "unique_argument_types": 15
  },
  "pair_statistics": {
    "total_pairs": <int>,
    "local_candidate_pairs": <int>,
    "distant_baseline_pairs": <int>
  },
  "semantic_similarity_analysis": {
    "distance_1": { "count": <int>, "mean": <float>, "median": <float>, ... },
    "distance_2": { ... },
    ...
    "distant_baseline": { ... }
  },
  "local_vs_distant": {
    "local_mean": <float>,
    "local_median": <float>,
    "local_count": <int>,
    "distant_mean": <float>,
    "distant_median": <float>,
    "distant_count": <int>,
    "mean_difference": <float>,
    "effect_size_cohens_d": <float>,
    "interpretation": "negligible|small|medium|large"
  },
  "actor_transitions": {
    "total_transitions": <int>,
    "transitions": [
      {
        "from": <str>,
        "to": <str>,
        "count": <int>,
        "percentage": <float>
      },
      ...
    ]
  },
  "argument_type_transitions": { ... },
  "combined_signal_analysis": {
    "methodology": { ... },
    "local_candidates": { "count": <int>, "mean": <float>, ... },
    "distant_baseline": { ... },
    "sample_high_scores": [...]
  },
  "limitations": [
    "No gold-standard passage relationship labels",
    ...
  ]
}
```

## Reproducibility

**Random Seed**: `RANDOM_SEED = 42`

Affects:
- Random pair sampling (distant pairs)
- Any randomization in analysis

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Deterministic encoding (same text always produces same embedding)

**Sorting**: All results are sorted by frequency/value (deterministic)

**Result**: Multiple runs with same data produce identical results

## Scalability

### Memory
- Embeddings: ~2.8 MB per 1000 passages (384-dim float32)
- Pair features: ~200 bytes per pair
- For 100k passages: ~280 MB embeddings + pairs

### Computation
- Encoding 10k passages: ~2-5 minutes (GPU accelerated when available)
- Pair construction: O(N × W) where W = local_window (typically O(N))
- Analysis: Linear in number of pairs

### Optimization
- Batch encoding (default 32)
- Distant pair sampling (reduces pair count ~85%)
- GPU acceleration (automatic with sentence-transformers)

## Important Notes

1. **Non-deterministic model loading**: First run downloads sentence-transformers model (~50MB)
2. **No hyperparameter tuning**: All parameters are fixed per specification
3. **No external datasets**: Uses only RMU:ECHR annotations provided
4. **Exploratory score**: Combined affinity is not validated, just illustrative
