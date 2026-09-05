# Gap 4 Experiment 1 Report

This exploratory experiment compares naturally ordered adjacent passages with multiple within-document random permutations.

- Documents used: 375
- Passages used: 31822
- Original adjacent pairs: 31447
- Shuffled sequences: 1875
- Original semantic mean / median: 0.570992 / 0.585081
- Shuffled semantic mean / median: 0.420295 / 0.418961
- Semantic mean difference: 0.150697
- Actor-continuity mean difference: 0.2578514485198634
- Statistical test: Wilcoxon signed-rank test on document-level mean semantic continuity (p=3.323478632207074e-63)
- Distinguishability at alpha=0.05: yes
- Evidence for H4: present for the semantic document-level comparison

## Interpretation

The result characterizes measurable continuity signals under an order-preserving condition and a disrupted control condition. It does not prove legal coherence detection, legal validity, or causal discourse structure. Any evidence supporting H4 must be described cautiously after inspecting the generated values.

Most common original argument-type transitions are available in `gap4_experiment_1_results.json`.
