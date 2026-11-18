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

### Current Indian dataset direction: LegalSeg

The research focus is shifting to Indian legal judgments. The selected resource is [L-NLProc/LegalSeg_CSV](https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV), associated with the paper [“LegalSeg: Unlocking the Structure of Indian Legal Judgments Through Rhetorical Role Classification” (Findings of NAACL 2025)](https://aclanthology.org/2025.findings-naacl.63/). The paper describes a large Indian judgment rhetorical-role dataset; the dataset repository lists `train.csv`, `val.csv`, and `test.csv`.

The planned CSV columns are `Index`, `Text`, and `Label`. Until the dataset files and their annotation methodology are inspected directly, README references to rows use **annotated text units**; they should not be assumed to be sentence-level rows. The role set to be examined is:

- Facts
- Issue
- Arguments of Petitioner (AoP)
- Arguments of Respondent (AoR)
- Reasoning
- Decision
- None

**LegalSeg is not included in this checkout, and no LegalSeg model has been trained.** The dataset files are access-gated on Hugging Face; therefore, this repository does not currently have verifiable CSV row counts, split statistics, or label frequencies. Those statistics are intentionally not reported here. Verify the paper’s annotation methodology and calculate counts and distributions from the actual train, validation, and test CSVs after obtaining access.

The planned LegalSeg sequence is:

1. Inspect and validate the official dataset files, schemas, labels, and annotation methodology.
2. Establish a majority-class baseline.
3. Train a TF-IDF + Logistic Regression baseline.
4. Evaluate macro-F1, per-class precision/recall/F1, and a confusion matrix.
5. Preserve the official train/validation/test split. Use validation—not test—for model selection or tuning; reserve test for final evaluation.
6. Consider transformer fine-tuning only if baseline results justify it and compute is available.

This work is **planned / in progress**, not completed.

## Current system architecture

The runtime implementation is in [`backend/app/`](backend/app/). The API currently performs the following:

1. **Upload and validation:** FastAPI accepts a multipart PDF upload, streams it to a generated filename, enforces a default 25 MiB limit, and checks for a PDF signature.
2. **Text extraction:** PyMuPDF extracts text page by page. Pages that appear scanned or image-heavy and contain little selectable text use a Tesseract OCR fallback. Extraction metadata records page-level method and basic quality indicators.
3. **Sentence and passage construction:** Extracted text (including page markers) is segmented by spaCy’s English sentencizer and grouped into passages of five sentences. Passage metadata contains order and character offsets in the extracted text.
4. **Semantic representation:** `sentence-transformers/all-MiniLM-L6-v2` creates 384-dimensional embeddings. In the runtime, embeddings support similarity, relationship, and coherence signals; they do **not** drive summary passage selection.
5. **Passage selection:** The summarizer groups passages into four-passage windows and, for longer documents, ranks windows using legal-marker matches and first/last-window bonuses. It selects a subset while avoiding adjacent windows, then restores source order. This can omit important content.
6. **Hierarchical abstractive summarization:** A Hugging Face summarization pipeline (default model `facebook/bart-large-cnn`) summarizes each selected group, reduces intermediate summaries in bounded batches, and generates a final summary. This is a general summarization model, not a model fine-tuned or validated for Indian judgments.
7. **Relationship and continuity analysis:** Metadata includes local passage similarity/distance signals and a comparison of adjacent similarity in original order versus deterministic shuffled orders. Optional RMU:ECHR classifier predictions are added only when a saved classifier artifact is available. These are exploratory signals, not validated legal relationships or Indian role labels.
8. **Response and source references:** The API returns the summary and metadata, including selected passage IDs and character offsets, generation trace, extraction information, and preliminary grounding checks. Offsets refer to extracted text, not clickable PDF page citations.

The source-overlap check filters generated sentences using lexical token overlap and identifier-presence rules. It is not entailment verification, fact-checking, or proof that a summary statement is supported in context.

## Implemented and planned capabilities

| Capability | Status |
| --- | --- |
| Upload a PDF and extract digital text | Implemented |
| OCR fallback for selected scanned/image-heavy pages | Implemented; requires the Tesseract executable and language data |
| Build source-ordered sentence-grouped passages and embeddings | Implemented |
| Select passage windows and generate hierarchical abstractive summaries | Implemented |
| Return passage IDs, extracted-text offsets, and analysis metadata | Implemented; references are not page-level citations |
| Compute exploratory semantic relationship and continuity indicators | Implemented; not evidence of legal coherence |
| Attach RMU:ECHR functional predictions when the optional model artifact exists | Implemented as cross-domain transfer only |
| Use LegalSeg labels to classify Indian rhetorical roles | Planned; no LegalSeg training completed |
| Structure output into facts, issues, each party’s arguments, reasoning, and decision | Intended product capability; not currently implemented as distinct validated sections |
| Provide a public-facing web interface | Planned; no frontend is present |
| Validate summary faithfulness with expert-annotated Indian judgments | Needed; not yet completed at sufficient scale |

## Limitations and research integrity

- RMU:ECHR annotations originate from European human-rights judgments, not Indian legal judgments.
- RMU:ECHR classifier predictions transferred to Indian documents are not Indian ground truth.
- Semantic similarity does not prove legal coherence, a relationship, or correctness.
- Lexical overlap checks do not establish factual entailment and may miss unsupported or contradictory claims.
- A general summarization model may omit, distort, or misattribute legal information.
- Passage selection is selective and may omit facts, arguments, qualifications, or reasoning that matter.
- OCR and sentence segmentation can introduce errors; source offsets refer to extracted text.
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
  services/                     Extraction, passages, embeddings, summarization,
                                and optional transferred classifier
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
```

### Configuration

The API reads these environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MAX_UPLOAD_BYTES` | Maximum upload size | `26214400` (25 MiB) |
| `UPLOAD_DIR` | Uploaded PDFs | `<project-root>/uploads` |
| `PROCESSED_DIR` | Extracted text saved by `/upload` | `data/processed` relative to the working directory |
| `TESSERACT_CMD` | Tesseract executable path, if not on `PATH` | Unset |
| `TESSERACT_LANG` | OCR language | `eng` |
| `OCR_DPI` | Page rendering resolution for OCR | `300` |
| `SUMMARIZATION_MODEL` | Hugging Face summarization model | `facebook/bart-large-cnn` |
| `RMU_CLASSIFIER_PATH` | Optional saved classifier artifact | `models/rmu_gap2_classifier.joblib` |
| `RMU_CLASSIFIER_CONFIG` | Optional classifier label/model configuration | `configs/rmu_gap2_classifier.json` |

The default uploads and model artifacts are ignored by Git. `/upload` retains its generated PDF and writes extracted text to `PROCESSED_DIR`. `/summarize` removes its temporary PDF after processing.

## API endpoints

All upload endpoints accept a multipart/form-data field named `file`.

| Method and path | Behavior |
| --- | --- |
| `GET /` | Basic service status |
| `POST /upload` | Extract PDF text, persist it under `PROCESSED_DIR`, and return extraction metadata and a preview |
| `POST /summarize` | Extract and summarize a PDF; return the summary and source/analysis metadata |

Example:

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
  -F "file=@path/to/judgment.pdf"
```

`/summarize` returns fields `summary` and `metadata`. Metadata can include extraction quality, passage counts and selected indices, extracted-text source offsets, generation trace, coverage measures, transferred functional predictions (if available), relationship/coherence signals, and lexical grounding-check results.

## Optional RMU:ECHR classifier

The API does not train a classifier at request time. If the RMU:ECHR annotations are present locally, the optional classifier can be trained with:

```bash
python -m backend.app.scripts.train_rmu_classifier
```

The script writes a model artifact under `models/` and classifier configuration under `configs/`. The tracked configuration file alone is not a trained model. Without the artifact, summarization continues without functional predictions. With it, predictions on Indian judgments remain transferred RMU:ECHR predictions, not Indian ground truth.

## Roadmap

1. Obtain and inspect LegalSeg; document its annotation unit, official split statistics, and label distribution from the actual CSV files and paper.
2. Establish majority and TF-IDF + Logistic Regression baselines with strict train/validation/test separation.
3. Decide whether transformer fine-tuning is justified by baseline results and available compute.
4. Develop evaluation for summary faithfulness, coverage, party attribution, reasoning, outcome, readability, and source traceability with expert input.
5. Compare against prompted general-purpose LLMs and suitable Indian legal summarization systems.
6. Investigate structure-aware output for facts, issues, party arguments, court reasoning, and decision, with links to the relevant original text.
7. Build a public-friendly platform only as evidence and evaluation support it.

## Further reading and onboarding

No `docs/PROJECT_CONTEXT.md`, `docs/RESEARCH_AND_EXPERIMENTS.md`, `docs/DATASETS_AND_STATUS.md`, `docs/ARCHITECTURE_AND_ROADMAP.md`, or `docs/AI_ASSISTANT_CONTEXT.md` files exist in this checkout. This README is the current onboarding entry point; the links below go only to documentation and outputs that are present:

- [Gap 1 implementation report](reports/GAP1_IMPLEMENTATION_REPORT.md)
- [Gap 3 experiment overview](backend/app/experiment_ip/gap3/README.md)
- [Gap 3 technical specification](backend/app/experiment_ip/gap3/TECHNICAL_SPEC.md)
- [Gap 4 experiment notes](backend/app/experiment_ip/gap4/README.md)
- [Gap 4 saved results](data/experiments/gap4/GAP4_EXPERIMENT_1_REPORT.md)
- [Runtime pipeline tests](tests/test_runtime_pipeline.py)
- [LegalSeg paper](https://aclanthology.org/2025.findings-naacl.63/)
- [LegalSeg dataset](https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV)
