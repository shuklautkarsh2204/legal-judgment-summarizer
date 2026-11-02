# Legal Judgment Summarizer

FastAPI service for extracting and hierarchically summarizing long legal judgment PDFs. Gap 1–4 research experiments remain separate from the uploaded-document runtime.

## Run From The Repository Root

Install the Python dependencies with `python -m pip install -r requirements.txt`. OCR also requires the Tesseract executable on the host. Start the API from this directory:

```bash
uvicorn backend.app.main:app --reload
```

Swagger UI is available at `http://127.0.0.1:8000/docs`. `POST /summarize` returns the summary, extraction quality, selected passage indices, source character offsets, coverage measures, and grounding limitations. `POST /upload` remains available for text extraction.

Uploads are limited to 25 MiB by default (`MAX_UPLOAD_BYTES`), stored under generated names, and removed after `/summarize` completes. `UPLOAD_DIR` and `PROCESSED_DIR` can override runtime directories. Set `TESSERACT_CMD`, `TESSERACT_LANG`, or `OCR_DPI` to configure OCR. Set `SUMMARIZATION_MODEL` to select a Hugging Face summarization model; the default is `facebook/bart-large-cnn`.

## Optional RMU Transfer Classifier

The API never trains the Gap 2 classifier during a request. To reproduce training and save an inference artifact, run from the repository root:

```bash
python -m backend.app.scripts.train_rmu_classifier
```

This uses the existing RMU:ECHR dataset and the original Gap 2 label/configuration in `configs/rmu_gap2_classifier.json`. The generated `models/rmu_gap2_classifier.joblib` is ignored by Git. Configure alternate paths with `RMU_CLASSIFIER_PATH` and `RMU_CLASSIFIER_CONFIG`. Without the artifact, summaries continue without classifier predictions.

RMU:ECHR functional labels applied to Indian judgments are transferred predictions, not Indian ground truth. The runtime does not import ECHR actor statistics. Adjacent semantic similarity and coherence indicators are exploratory and have not been validated on Indian judgments.

## Grounding And Evaluation

Generation is hierarchical: selected, source-ordered passage groups produce intermediate summaries, which are reduced in bounded batches with source references retained. Generated sentences that fail lexical-overlap or identifier-presence checks are excluded and reported. These checks are not legal entailment verification and cannot reliably detect contradictions; outputs require human legal review. Coverage and semantic continuity values are computed from the uploaded document, not presented as gold-standard evaluation scores.
