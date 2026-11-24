# Indian Legal Judgment Understanding and Summarization

This repository is a research prototype for making Indian court judgments easier to navigate and understand. It combines an implemented PDF-processing and summarization API with exploratory Legal NLP experiments. The longer-term research goal is to represent legal structure and relationships—not merely shorten a document—and to make summaries traceable to their source.

> **Important:** This system does not provide legal advice and does not replace lawyers or the original judgment. Generated summaries and analysis may be incomplete or wrong. The original judgment remains authoritative.

## Why this project

Indian judgments can be lengthy, structurally complex, and written in specialized legal language. Citizens, students, journalists, researchers, and legal professionals may find it difficult to quickly locate the facts, issues, party submissions, reasoning, and outcome that matter to them.

We aim to investigate NLP methods that can produce concise, coherent, plain-language explanations while preserving important legal information and helping readers trace statements back to the judgment. The intended product is more than a generic PDF summarizer: its central research question is whether understanding legal structure and reasoning can make summaries more useful and faithful.

## Research questions

A systematic mapping study of approximately 75 Legal NLP papers informed our research agenda. The foundation includes [“Legal Natural Language Processing From 2015 to 2022: A Comprehensive Systematic Mapping Study of Advances and Applications” (IEEE Access)](https://doi.org/10.1109/ACCESS.2023.3333946).

The study motivated four open research questions. They are questions to investigate, not capabilities this project has already solved:

1. **Long-document representation:** How can lengthy legal documents be represented and processed without losing important information, context, or source location?
2. **Legal-functional understanding:** How can the different types and functions of legal content—such as facts, arguments, reasoning, and decisions—be identified?
3. **Passage relationships:** How can relationships between different parts of a judgment be identified, including material separated by many pages?
4. **Discourse coherence:** How can the coherence and progression of legal discourse be characterized, and what signals distinguish the original progression from disrupted sequences?

## Research history and findings

The initial experiments used RMU:ECHR, a dataset of annotations from the **European Court of Human Rights**. It is not Indian legal data. Its labels and model predictions must not be treated as Indian legal ground truth.

| Direction | Work and evidence in this repository | Interpretation |
| --- | --- | --- |
| **Gap 1 — long-document representation** | Five-sentence passages and 384-dimensional `all-MiniLM-L6-v2` embeddings were produced for the Indian judgment `Vijay_Madanlal_Choudhary_vs_Union_Of_India_on_27_July_2022`. The saved statistics report 4,562 sentences, 913 passages, 124,922 words, 764,313 characters, and a valid passage-order check. The recorded processing time was about 142 seconds. | This demonstrates passage construction, embedding, and traceability mechanics on one Indian judgment. It does not validate legal roles, passage relationships, or summary quality. |
| **Gap 2 — legal-functional classification** | An RMU:ECHR Logistic Regression classifier used sentence-transformer embeddings and four selected argument-type labels. In the saved document-level split results, it achieved 0.7327 accuracy and 0.7323 macro-F1, with 250 training documents and 63 test documents. | These are results on the RMU:ECHR task and labels, not Indian legal classification results. The classifier artifact is optional and its predictions on Indian judgments are transferred predictions, not Indian annotations. |
| **Gap 3 — passage relationships** | The experiment code explores semantic similarity, distance/proximity, and annotation-derived actor and argument-type information. The experiment documentation explicitly treats these as descriptive signals, not gold relationship labels. | Relationship signals do not demonstrate that two passages form a legally meaningful argument-response relation. The required original RMU:ECHR document texts are not included in this repository, and no Gap 3 result file is present in the checkout. |
| **Gap 4 — discourse continuity** | The saved RMU:ECHR experiment compared original passage order with five within-document shuffled sequences for each of 375 documents (31,822 passages). Mean adjacent semantic similarity was 0.5710 in original order and 0.4203 in shuffled sequences; the mean difference was 0.1507. A document-level paired Wilcoxon test reported *p* = 3.32 × 10⁻⁶³. | This is evidence that this embedding-based continuity signal differs between original and shuffled sequences in this RMU:ECHR sample. It does not prove legal coherence, correctness, or reasoning understanding, and it does not validate the signal on Indian judgments. |

Experiment details and source outputs are in [`backend/app/experiment_ip/`](backend/app/experiment_ip/) and [`data/experiments/`](data/experiments/). These numbers describe saved experiments, not production quality or comparative performance. They should be revisited alongside the scripts and their documented caveats.

### Exploratory transfer to an Indian judgment

The repository includes an exploratory transfer test on `Vijay_Madanlal_Choudhary_vs_Union_Of_India_on_27_July_2022`. It applies a classifier trained on RMU:ECHR labels to passages from that Indian judgment. This single-document transfer test is not evidence of generalization to Indian judgments: it has no Indian gold labels and does not establish prediction correctness.

### Indian role-label dataset direction: paused pending access

The candidate resource is [L-NLProc/LegalSeg_CSV](https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV), associated with [“LegalSeg: Unlocking the Structure of Indian Legal Judgments Through Rhetorical Role Classification” (Findings of NAACL 2025)](https://aclanthology.org/2025.findings-naacl.63/). Access is currently unreliable. **No runtime feature or next development milestone depends on downloading LegalSeg.**

The planned CSV columns are `Index`, `Text`, and `Label`. Until the dataset files and their annotation methodology are inspected directly, README references to rows use **annotated text units**; they should not be assumed to be sentence-level rows. The role set to be examined is:

- Facts
- Issue
- Arguments of Petitioner (AoP)
- Arguments of Respondent (AoR)
- Reasoning
- Decision
- None

**LegalSeg is not included in this checkout, and no LegalSeg model has been trained.** This repository has no verifiable CSV row counts, split statistics, or label frequencies. Those statistics are intentionally not reported here. Do not fabricate Indian rhetorical-role annotations or claim an Indian-trained classifier.

If an accessible, suitable dataset or manually validated annotation set becomes available, a future optional study should:

1. Inspect and validate the official dataset files, schemas, labels, and annotation methodology.
2. Establish a majority-class baseline.
3. Train a TF-IDF + Logistic Regression baseline.
4. Evaluate macro-F1, per-class precision/recall/F1, and a confusion matrix.
5. Preserve the official train/validation/test split. Use validation—not test—for model selection or tuning; reserve test for final evaluation.
6. Consider transformer fine-tuning only if baseline results justify it and compute is available.

Until then, RMU:ECHR predictions remain optional, exploratory transfer signals. The explicit-text role cues in the runtime are transparent heuristics, not an Indian-trained classifier.

## Current system architecture

The runtime implementation is in [`backend/app/`](backend/app/). The API currently performs the following:

1. **Upload and validation:** FastAPI accepts a multipart PDF upload, streams it to a generated filename, enforces a default 25 MiB limit, and checks for a PDF signature.
2. **Text extraction:** PyMuPDF extracts text page by page. Pages that appear scanned or image-heavy and contain little selectable text use a Tesseract OCR fallback. Extraction metadata records page-level method, quality indicators, and character ranges for page text.
3. **Sentence and passage construction:** Extracted text (including page markers) is segmented by spaCy’s English sentencizer and grouped into passages of five sentences. Passage metadata contains order and character offsets in the extracted text; page references are attached where extracted page ranges overlap.
4. **Semantic representation:** `sentence-transformers/all-MiniLM-L6-v2` creates 384-dimensional embeddings. In the runtime, embeddings support similarity, relationship, and coherence signals; they do **not** drive summary passage selection.
5. **Role-aware evidence organization:** A separate module assigns only unambiguous explicit textual cues to party arguments, court reasoning/decision, facts/procedure, issues, and precedents. Ambiguous passages remain unclassified. Optional RMU:ECHR labels receive an explicitly approximate functional mapping; they never assign a speaker or override uncertain text cues.
6. **Passage selection:** The summarizer groups passages into four-passage windows and, for longer documents, ranks windows using legal-marker matches and first/last-window bonuses. It selects a subset while avoiding adjacent windows, restores source order, and reports selected and omitted passage IDs plus per-window scores/reasons. The policy can still omit important content.
7. **Hierarchical abstractive summarization:** A Hugging Face summarization pipeline (default model `facebook/bart-large-cnn`) summarizes role-tagged selected groups, reduces intermediate summaries in bounded batches, and generates a final summary. The model input limit is read from model/tokenizer metadata when available; oversized input is split without enabling truncation. This is a general summarization model, not one fine-tuned or validated for Indian judgments.
8. **Claim-level evidence checks:** Each generated claim in the generation trace and each returned claim receives source passage IDs, document ID, offsets, available pages, an explicit evidence role, a heuristic verification result, and an uncalibrated lexical-overlap score. Claims without role-compatible identified evidence are withheld from the returned summary and remain visible in verification metadata for review. Role problems, possible contradictions, insufficient evidence, and human-review abstentions are distinct outcomes. This is not NLI or entailment verification.
9. **Structured view and research signals:** The response can include claim groups for facts, issues, party arguments, court reasoning, decisions, and precedents only when role-compatible evidence is identified. Existing passage relationship/coherence signals and the saved Gap 4 experiment are retained as exploratory measures, not proof of legal reasoning.

The existing source-overlap check is still a lexical heuristic; it is not entailment, fact-checking, or proof of contextual support. Paragraph identifiers and bounding boxes are not currently reliable in the extraction layer and are marked unavailable rather than invented.

```text
Indian Judgment PDF
  -> PDF/OCR extraction with page text ranges
  -> ordered passages, offsets, and embeddings
  -> explicit-cue role organization
       + optional, separately labeled RMU:ECHR transfer
  -> auditable selection (selected and omitted passage IDs)
  -> token-bounded, hierarchical role-aware generation
  -> per-claim lexical evidence/role/contradiction checks
  -> traceable summary + evidence-backed structured sections
```

## Implemented and planned capabilities

| Capability | Status |
| --- | --- |
| Upload a PDF and extract digital text | Implemented |
| OCR fallback for selected scanned/image-heavy pages | Implemented; requires the Tesseract executable and language data |
| Build source-ordered sentence-grouped passages and embeddings | Implemented |
| Assign transparent explicit-text role cues and keep ambiguous content uncertain | Implemented as heuristic; not validated Indian role classification |
| Select passage windows and report selected/omitted passages | Implemented; selection remains heuristic and lossy |
| Generate hierarchical abstractive summaries without silent tokenizer truncation | Implemented using available model limits; incomplete model metadata remains a caveat |
| Verify generated claims individually and retain evidence references | Implemented as lexical/role heuristics, not entailment or legal accuracy |
| Return passage IDs, extracted-text offsets, and available page references | Implemented; paragraph/bounding-box references unavailable |
| Compute exploratory semantic relationship and continuity indicators | Implemented; not evidence of legal coherence |
| Attach RMU:ECHR functional predictions when the optional model artifact exists | Implemented as cross-domain transfer only |
| Use LegalSeg labels to classify Indian rhetorical roles | Paused pending reliable access; no LegalSeg training completed |
| Structure output into evidence-backed legal sections | Implemented as generated claim groups when role-compatible evidence is identified; not validated |
| Provide a developer Streamlit interface for PDF upload, summaries, verification, and provenance | Implemented in `frontend/`; requires the existing FastAPI backend |
| Validate summary faithfulness with expert-annotated Indian judgments | Needed; not yet completed at sufficient scale |

## Limitations and research integrity

- RMU:ECHR annotations originate from European human-rights judgments, not Indian legal judgments.
- RMU:ECHR classifier predictions transferred to Indian documents are not Indian ground truth.
- Semantic similarity does not prove legal coherence, a relationship, or correctness.
- Lexical overlap checks do not establish factual entailment and may miss unsupported or contradictory claims.
- Role cues and contradiction terms are deliberately small heuristics; ambiguous passages and claims require human review.
- Verification overlap scores are exploratory and uncalibrated. No thresholds have been calibrated on Indian legal annotations.
- No NLI model is currently integrated. General NLI scores would be signals to validate, not proof of legal correctness; legal stance, speaker, and cross-passage context require domain-specific evaluation.
- A general summarization model may omit, distort, or misattribute legal information.
- Passage selection is selective and may omit facts, arguments, qualifications, or reasoning that matter.
- OCR and sentence segmentation can introduce errors; source offsets refer to extracted text, not PDF glyph coordinates.
- Page references are available only when page-range metadata is present. Paragraph references and bounding boxes are explicitly unavailable.
- `/upload` and `/summarize` delete temporary PDFs after processing. Extracted text is transient by default; `/upload` persists it only when `RETAIN_PROCESSED_TEXT=true`.
- CPU-heavy extraction and inference run in FastAPI's worker thread pool; this avoids synchronous work on the event loop but does not add request cancellation or resource isolation.
- The runtime and transfer experiment have not been validated through a sufficiently large, expert-annotated Indian legal evaluation.
- A technically working API does not establish research effectiveness, legal reliability, or user benefit.
- This prototype is not legal advice. The original judgment remains authoritative.

## Evaluation direction

The system should be evaluated against general-purpose LLMs prompted to summarize the same judgment and, where suitable and reproducible, existing Indian legal summarization approaches. Reference summaries and expert review should be used where available. Comparative claims must wait for controlled experiments.

Potential evaluation dimensions include:

- Factual faithfulness and unsupported claims
- Coverage of important legal information
- Correct attribution of petitioner and respondent arguments
- Preservation of the court’s reasoning
- Accuracy of the final decision
- Coherence and readability
- Traceability of summary statements to relevant original passages

Summarization quality should be evaluated separately from the LegalSeg rhetorical-role classification task. For classification, preserve the official split and report per-class performance, not macro averages alone.

## Technology

- **Language and API:** Python, FastAPI, Uvicorn
- **PDF and OCR:** PyMuPDF, pytesseract, Tesseract OCR executable, Pillow
- **Segmentation and embeddings:** spaCy sentencizer, Sentence Transformers (`all-MiniLM-L6-v2`)
- **Summarization:** Hugging Face Transformers (`facebook/bart-large-cnn` by default)
- **Classification and numerical analysis:** scikit-learn, NumPy
- **Experiments and plots:** SciPy, Matplotlib, JSON, and CSV

## Repository map

```text
backend/app/
  main.py                       FastAPI routes and upload handling
  services/                     Extraction, passages, embeddings, role-aware
                                evidence checks, summarization, and transfer classifier
  scripts/                      RMU:ECHR classifier artifact training
  experiment_ip/gap1..gap4/     Research experiment code and experiment notes
configs/
  rmu_gap2_classifier.json      Configuration for optional RMU:ECHR classifier
data/experiments/
  gap1/                         Indian judgment passage artifacts and statistics
  gap2/                         RMU:ECHR annotations and classifier outputs
  gap4/                         RMU:ECHR continuity results and plots
reports/                        Experiment and implementation reports
tests/
  test_runtime_pipeline.py      Runtime API and pipeline tests
  test_evidence_verification.py Synthetic role and claim-verification cases
requirements.txt                Python package requirements
README.md                       Project onboarding and current status
```

LegalSeg CSVs, the optional trained classifier artifact, uploads, and processed runtime text are not committed. Gap 3’s original RMU:ECHR XMI document texts are also external to this checkout.

## Setup and run

Use an environment compatible with the packages in [`requirements.txt`](requirements.txt). Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Install the **Tesseract OCR executable** separately if scanned-page OCR is needed. The Python wrapper alone does not install it. The application uses the English OCR language by default. The first request that needs embeddings or summarization may download model files from Hugging Face.

Start the API from the repository root:

```bash
uvicorn backend.app.main:app --reload
```

The service listens at `http://127.0.0.1:8000`. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### Run the runtime tests

From the repository root:

```bash
python -m unittest discover -s tests -p "test_runtime_pipeline.py"
python -m unittest discover -s tests -p "test_evidence_verification.py"
```

### Configuration

The API reads these environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MAX_UPLOAD_BYTES` | Maximum upload size | `26214400` (25 MiB) |
| `UPLOAD_DIR` | Uploaded PDFs | `<project-root>/uploads` |
| `PROCESSED_DIR` | Destination for extracted text when persistence is enabled | `data/processed` relative to the working directory |
| `RETAIN_PROCESSED_TEXT` | Persist `/upload` extracted text (`true`/`false`) | `false` |
| `TESSERACT_CMD` | Tesseract executable path, if not on `PATH` | Unset |
| `TESSERACT_LANG` | OCR language | `eng` |
| `OCR_DPI` | Page rendering resolution for OCR | `300` |
| `SUMMARIZATION_MODEL` | Hugging Face summarization model | `facebook/bart-large-cnn` |
| `RMU_CLASSIFIER_PATH` | Optional saved classifier artifact | `models/rmu_gap2_classifier.joblib` |
| `RMU_CLASSIFIER_CONFIG` | Optional classifier label/model configuration | `configs/rmu_gap2_classifier.json` |

Both endpoints delete their temporary PDF after processing. `/upload` returns `storage: "transient"` and no text-file path by default; set `RETAIN_PROCESSED_TEXT=true` to preserve extracted text under `PROCESSED_DIR`. `/summarize` does not persist extracted text. The `/upload` response still includes a short preview for the requesting client. Application logs do not include document text or generated summaries.

## API endpoints

All upload endpoints accept a multipart/form-data field named `file`.

| Method and path | Behavior |
| --- | --- |
| `GET /` | Basic service status |
| `POST /upload` | Extract PDF text and return extraction metadata and a preview; extracted text is transient unless retention is explicitly enabled |
| `POST /summarize` | Extract and summarize a PDF; return the summary and source/analysis metadata |

Example:

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
  -F "file=@path/to/judgment.pdf"
```

`/summarize` returns fields `summary` and `metadata`. Metadata can include extraction quality, page-aware source offsets, selected and omitted passage IDs, generation trace with per-claim checks, evidence-backed structured sections, uncalibrated role/overlap outcomes, transferred functional predictions (if available), relationship/coherence signals, and lexical grounding-check results. Existing clients can continue reading `summary`; the metadata object has additive fields.

## Streamlit developer frontend

Install the frontend dependencies and start the app from the repository root:

```bash
python -m pip install -r frontend/requirements.txt
python -m streamlit run frontend/app.py
```

The frontend sends uploaded PDFs to `POST /summarize` as multipart form data in the `file` field and waits for the synchronous response. Set `BACKEND_URL` to use a backend other than `http://localhost:8000`. The request read timeout defaults to 1800 seconds and can be changed with `BACKEND_TIMEOUT_SECONDS`.

The interface displays the summary, only the structured sections returned by the backend, claim-level verification, and available source passage/page/offset references. The backend does not currently return source passage text or PDF page images as part of those references. Verification scores are backend-provided heuristic signals, not proof of legal correctness.

## Optional RMU:ECHR classifier

The API does not train a classifier at request time. If the RMU:ECHR annotations are present locally, the optional classifier can be trained with:

```bash
python -m backend.app.scripts.train_rmu_classifier
```

The script writes a model artifact under `models/` and classifier configuration under `configs/`. The tracked configuration file alone is not a trained model. Without the artifact, summarization continues without functional predictions. With it, predictions on Indian judgments remain transferred RMU:ECHR predictions, not Indian ground truth.

## Roadmap

1. Prepare an expert-reviewed Indian judgment validation set and calibrate role/claim-verification thresholds; synthetic tests are not accuracy results.
2. Compare (a) a general LLM prompted on the judgment, (b) the existing summarization pipeline, and (c) the role-aware evidence-grounded pipeline.
3. Measure factual accuracy, party attribution, preservation of reasoning, outcome/direction accuracy, coverage, contradiction rate, traceability, and plain-language readability.
4. Evaluate role heuristics, lexical grounding, any future NLI adapter, and end-to-end faithfulness as separate tasks. Use NLI only as an uncalibrated signal until validated with legal claims.
5. Resume the optional LegalSeg baselines only when reliable dataset access and annotation documentation are available; retain the official split and report per-class metrics.
6. Consider transformer fine-tuning only if baseline results justify it and compute is available.
7. Build a public-facing interface only as evidence and evaluation support it.

## Further reading and onboarding

No `docs/PROJECT_CONTEXT.md`, `docs/RESEARCH_AND_EXPERIMENTS.md`, `docs/DATASETS_AND_STATUS.md`, `docs/ARCHITECTURE_AND_ROADMAP.md`, or `docs/AI_ASSISTANT_CONTEXT.md` files exist in this checkout. This README is the current onboarding entry point; the links below go only to documentation and outputs that are present:

- [Gap 1 implementation report](reports/GAP1_IMPLEMENTATION_REPORT.md)
- [Gap 3 experiment overview](backend/app/experiment_ip/gap3/README.md)
- [Gap 3 technical specification](backend/app/experiment_ip/gap3/TECHNICAL_SPEC.md)
- [Gap 4 experiment notes](backend/app/experiment_ip/gap4/README.md)
- [Gap 4 saved results](data/experiments/gap4/GAP4_EXPERIMENT_1_REPORT.md)
- [Runtime pipeline tests](tests/test_runtime_pipeline.py)
- [Role-aware pipeline audit and evaluation plan](reports/ROLE_AWARE_PIPELINE.md)
- [Synthetic claim-verification tests](tests/test_evidence_verification.py)
- [LegalSeg paper](https://aclanthology.org/2025.findings-naacl.63/)
- [LegalSeg dataset](https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV)
