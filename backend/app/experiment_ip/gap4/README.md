# Gap 4 Experiment 1

This experiment tests whether naturally ordered adjacent passages show different measurable continuity signals from within-document random permutations.

It is a separate experiment from Gap 3 and does not modify Gap 3 code or outputs. It reuses Gap 3's corrected original RMU:ECHR document loader, full annotation validation, five-sentence passage construction, original character coordinates, embeddings, actor sets, argument-type sets, and deterministic primary labels.

## Run

From the project root:

```powershell
& .\.venv\Scripts\python.exe backend/app/experiment_ip/gap4/experiment_1_coherence.py
```

The configured environment must have the dependencies in `requirements.txt`, including spaCy, sentence-transformers, SciPy, and matplotlib.

## Method

- Uses every loaded document with at least two passages.
- Computes semantic similarity, actor-set overlap, and actor/argument-type transitions for original adjacent pairs.
- Generates five seeded random permutations per document without moving passages between documents.
- Uses the same signal calculations for each disrupted sequence.
- Uses document-level paired means for the Wilcoxon signed-rank comparison.
- Treats permutations as a control condition, not as proven incorrect legal reasoning.

## Outputs

Outputs are written only to `data/experiments/gap4/`:

- `gap4_experiment_1_results.json`
- `gap4_experiment_1_raw_sequences.json`
- `gap4_experiment_1_configuration.json`
- `gap4_document_level_statistics.csv`
- semantic continuity and argument-transition PNG plots
- `GAP4_EXPERIMENT_1_REPORT.md`

The experiment is exploratory. RMU:ECHR labels are related-domain legal-functional annotations, not Indian Supreme Court relationship gold labels, and the results do not prove legal coherence detection.
