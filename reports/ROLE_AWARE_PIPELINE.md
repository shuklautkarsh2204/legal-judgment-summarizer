# Role-Aware Pipeline Audit and Implementation

## Scope and evidence standard

This is a repository-grounded audit and a selective integration of ideas from the
provided NyayaVerify design. NyayaVerify's reported implementation and performance
were not independently tested here. Existing experiment artifacts and their
methodology were preserved.

## Phase 1 — Audit of the original project

### Existing architecture

Before these changes, the runtime path was:

1. FastAPI streamed a PDF to an upload file, extracted page text with PyMuPDF and
   selectively used Tesseract for sparse/image-heavy pages.
2. spaCy segmented the extracted text into sentences; `PassageBuilder` grouped
   five sentences and retained passage order and extracted-text character offsets.
3. Sentence-transformer embeddings drove relationship/continuity metadata, not
   summary selection.
4. The summarizer selected four-passage windows using legal-marker matches and
   first/last-window bonuses. Longer documents could lose unselected passages;
   the old response did not explain omissions.
5. General-purpose `facebook/bart-large-cnn` generated hierarchical summaries.
   The pipeline passed `truncation=True`, so oversized inputs could be silently
   shortened.
6. A token-overlap and identifier-presence heuristic filtered generated text.
   It was not entailment verification and did not distinguish speakers.
7. Optional transferred RMU:ECHR predictions, local similarity pairs, and the
   ordered-versus-shuffled continuity signal were reported as exploratory.

The original API also performed extraction and model inference directly inside an
`async` route, printed the complete final summary, retained `/upload` PDFs, and
persisted extracted text by default. `/summarize` already removed its temporary
PDF in a `finally` block.

### Implemented behavior versus research evidence

| Area | Evidence in this checkout | Status and limitation |
| --- | --- | --- |
| Gap 1: long-document representation | Saved Indian judgment artifact contains 913 five-sentence passages from 4,562 sentences, with 764,313 extracted characters and a successful order check. | Passage construction/order/offset mechanics were demonstrated on this artifact. This is not a legal-role or summary-faithfulness evaluation. |
| Gap 2: functional labels | Saved RMU:ECHR document-level results report accuracy 0.7327 and macro-F1 0.7323 for four selected labels, with document-level train/test separation. | These are RMU:ECHR results only. The optional transferred predictions have no Indian gold labels and are not Indian ground truth. |
| Gap 2 transfer | Transfer experiment code and runtime integration exist for an Indian Supreme Court judgment. | Exploratory single-document domain transfer; no validated Indian classifier or accuracy result. |
| Gap 3: passage relationships | Experiment code and technical documentation exist. The original XMI documents are external to this checkout and no Gap 3 results file is present. Older Gap 3 notes conflict about synthetic versus original text. | Treat as implemented exploratory code, not as a completed, currently reproducible result. Do not compare or reinterpret historical values without the source documents and exact run. |
| Gap 4: discourse continuity | Saved experiment uses 375 RMU:ECHR documents, 31,822 passages, five within-document shuffles, and a document-level Wilcoxon comparison. Saved means are 0.5710 ordered and 0.4203 shuffled (difference 0.1507; p = 3.32e-63). | A semantic continuity signal differed from shuffled controls in this RMU:ECHR sample. This does not establish legal coherence or reasoning correctness. Original outputs were not changed. |
| Frontend | No frontend files or frontend tests are present. | Planned, not implemented. |
| Indian legal validation | No expert-annotated Indian claim/role evaluation set is present. | No legal-accuracy claim is supported. |

### Pre-change tests

Before editing, `python -m unittest discover -s tests -p
'test_runtime_pipeline.py' -v` passed all 8 tests. `python -m pytest -q` could
not run because `pytest` is not installed; the documented `unittest` runner was
used instead. The saved Gap 4 output and experiment settings were inspected; no
new experiment was run and no historical metrics were changed.

## NyayaVerify comparison

| Design idea | Existing partial support | Integration decision |
| --- | --- | --- |
| Separate party arguments from court findings | None in the original runtime. RMU:ECHR argument-type predictions are not speaker labels. | Add a separate, explicit-cue evidence organizer. Ambiguous or mixed passages stay unclassified. Party evidence cannot establish a court decision in the verifier. |
| Map existing legal-functional labels | Optional transferred RMU:ECHR predictions exist. | Map only selected German labels to approximate functions (precedents, reasoning, decision), mark the mapping cross-domain and exploratory, and never treat it as speaker attribution or Indian truth. |
| Claim-level verification | Existing overlap filtering is sentence-level but not role-aware. | Add per-generated-claim references and transparent overlap/role/contradiction outcomes. Keep overlap explicitly labeled as a heuristic. No entailment model or calibrated threshold is claimed. |
| Stronger traceability | Passage IDs and extracted-text offsets already exist. | Preserve source passage IDs and document IDs through every hierarchy level; add page references where extraction offsets establish them. Paragraph IDs and bounding boxes remain unavailable. |
| Structured legal summary | No separate legal sections existed. | Group returned claims into a section only when the role-aware heuristic identifies supporting passage evidence. The legacy `summary` string remains present. |
| Privacy and asynchronous work | Summary text was printed; uploaded PDFs and extracted text were retained by `/upload`; inference blocked the event loop. | Remove content logging, delete both endpoints' temporary PDFs, make extracted-text retention opt-in, sanitize error responses, and move extraction/inference to FastAPI's worker thread pool. |
| NLI | No NLI model or Indian validation data exists. | Do not add a model/dependency or thresholds. NLI could be tested later as a claim/passage signal, but generic NLI does not resolve speaker attribution, legal context, or multi-source reasoning by itself. |

## Phase 2 — Minimal design and provenance

```text
Indian Judgment PDF
  -> PDF/OCR extraction + page character ranges
  -> ordered passages + offsets + embeddings
  -> explicit-cue role organization
       + optional, separately labeled RMU:ECHR function mapping
  -> auditable selection (selected and omitted passage IDs)
  -> token-bounded hierarchical generation
  -> claim-level lexical evidence and role checks
  -> summary, evidence-backed sections, and trace references
```

`RoleAwareEvidenceOrganizer` is independent of generation. It outputs an evidence
role, assignment status, and method; it does not convert an RMU:ECHR prediction
into a speaker. `verify_claim` accepts one claim and source passages and returns
a role category, supporting IDs/references, a result, a method, a raw exploratory
score, and abstention reasons. NLI can be added behind this module after a
validation protocol exists.

Character offsets remain positions in the extracted text string, not PDF glyph
coordinates. Page numbers are attached by overlap with page-body character
ranges. Paragraph references and bounding boxes are returned as unavailable,
not inferred.

## Phase 3 — Implemented changes

### Role-aware evidence and claim-level output

- Added explicit textual-cue categories for facts/procedure, issues, petitioner
  arguments, respondent arguments, court reasoning, court decisions, precedents,
  and unclassified/ambiguous content.
- Generation receives role-tagged source passages and an instruction that party
  submissions must remain party-attributed.
- Each generated claim at each hierarchy level now carries evidence checks and
  source passage references. The returned summary withholds claims that do not
  receive role-compatible identified evidence; those claims remain available in
  verification metadata for review. Only passing claims populate
  `structured_summary`.
- Outcomes are distinct: `supported_by_identified_evidence`,
  `potential_role_attribution_problem`, `insufficient_evidence`,
  `possible_contradiction`, and `requires_human_review`.
- The score is lexical token coverage, is uncalibrated, and is explicitly not an
  entailment confidence. Contradiction cues cover a small, transparent set of
  opposing terms; they are not a complete contradiction detector.

### Selection, generation limits, and traceability

- Selection metadata includes every selected and omitted passage ID, each
  window's score, and the current omission constraint.
- Model/tokenizer limits are inspected when available. Oversized source input is
  split into ordered chunks; Transformers truncation is disabled. A source
  fragment that cannot be safely split raises an error rather than being
  silently shortened. Custom generators that expose no tokenizer limit are
  reported as having an unavailable limit and receive no truncation request.
- Source document IDs, passage IDs, offsets, and available page numbers are
  attached to claim evidence. Paragraph and bounding-box fields are explicitly
  unavailable.
- Existing relationship/coherence calculations and Gap 4 outputs were retained.

### Privacy, API, and execution

- No full judgment text or generated summary is printed or logged.
- Temporary PDFs are removed after `/upload` and `/summarize`; extracted text is
  not persisted by `/summarize`.
- `/upload` text persistence is now opt-in via `RETAIN_PROCESSED_TEXT=true`.
  The response keeps the `text_file` field but it is `null` by default and adds a
  `storage` field. The short response preview remains visible to the requesting
  client.
- PDF extraction and summarization run in FastAPI's thread pool instead of
  blocking the async event loop.
- Error responses no longer echo arbitrary exception messages that could
  contain document text.

### Files changed

- Runtime: [`main.py`](../backend/app/main.py),
  [`text_extraction.py`](../backend/app/services/text_extraction.py),
  [`summarization.py`](../backend/app/services/summarization.py), and the new
  [`evidence_verification.py`](../backend/app/services/evidence_verification.py).
- Tests: [`test_runtime_pipeline.py`](../tests/test_runtime_pipeline.py) and
  the new [`test_evidence_verification.py`](../tests/test_evidence_verification.py).
- Documentation: [`README.md`](../README.md) and this audit/report.
- No research data, Gap 1–4 artifacts, model artifacts, or experiment methods
  were modified.

## Phase 4 — Tests and interpretation

The synthetic unit tests exercise wrong court/party attribution, respondent to
petitioner misattribution, a supported court decision, insufficient evidence,
multi-passage support, possible contradiction, ambiguous roles, missing page
and paragraph metadata, cross-domain label mapping, and generation input
chunking. They test software behavior only; they are not legal-accuracy,
NLI-performance, or end-to-end-faithfulness results.
Generation-limit tests use a synthetic tokenizer and generator. No production
summarization model was downloaded or benchmarked, and no summary-quality claim
is made.

Final validation: `python -m unittest discover -s tests -v` passed all 19 tests.
`git diff --check` passed. The pre-change documented runtime suite passed all 8
tests; the full post-change suite includes those 8 plus 11 synthetic verification
and token-limit tests. `pytest` is not installed in the environment, so the
repository's documented `unittest` runner was used. Automated tests do not
replace expert review or validate heuristic thresholds.

## Evaluation plan and future research

1. Build an Indian judgment evaluation set with source passage offsets and
   expert-reviewed claim categories/speakers. Use separate documents for
   development and final test; record annotation disagreements and adjudication.
2. Compare the same judgments using: (a) a general LLM prompt, (b) the existing
   summarizer without role-aware evidence, and (c) the role-aware,
   evidence-grounded pipeline.
3. Measure factual correctness, petitioner/respondent attribution, preservation
   of judicial reasoning, decision/direction accuracy, coverage, contradiction
   rate, evidence traceability, and non-lawyer readability. Use blinded human
   review and legal-expert assessment for legal dimensions.
4. Evaluate role heuristics, lexical grounding, any future NLI component, and
   end-to-end faithfulness separately. Calibrate thresholds only on a suitable
   development set and report abstention/coverage alongside error rates.
5. Resume the optional LegalSeg baselines only after dataset access and its
   annotation methodology can be checked. Preserve its official splits; do not
   treat ECHR labels as Indian labels.
6. Reassess selective passage coverage and token chunking on representative
   long judgments before claiming improved summary quality.

Statutory concordance engines, limitation calculations, case-memory graphs,
messaging integrations, TTS, cloud GPU workers, large-scale ingestion, and
air-gapped deployments remain deferred: the checkout contains no validated
need or data for those components.
