import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend.app import main
from backend.app.services import text_extraction


ROOT = Path(__file__).resolve().parents[1]


class FakeEmbeddingAnalyzer:
    def encode_passages(self, passages, batch_size=32):
        encoded = []
        for passage in passages:
            item = passage.copy()
            embedding = [0.0] * 384
            embedding[0] = 1.0
            item["embedding"] = embedding
            encoded.append(item)
        return {"passages": encoded, "stats": {"total_passages": len(encoded)}}


class FakeGenerator:
    def __call__(self, source_text, **kwargs):
        if "SOURCE PASSAGES:\n" in source_text:
            source_text = source_text.split("SOURCE PASSAGES:\n", 1)[1]
        if "references [" in source_text.lower() or "sources [" in source_text.lower():
            match = re.search(r"\]:\s*(.+)", source_text)
            summary = match.group(1).splitlines()[0] if match else source_text
        else:
            summary = re.split(r"(?<=[.!?])\s+", source_text.strip(), maxsplit=1)[0]
        return [{"summary_text": summary.strip()}]


def create_pdf(path, text, page_chunk_size=1600):
    document = pymupdf.open()
    for offset in range(0, len(text), page_chunk_size):
        page = document.new_page(width=595, height=842)
        chunk = text[offset:offset + page_chunk_size]
        page.insert_textbox(
            pymupdf.Rect(40, 35, 555, 805),
            chunk,
            fontsize=7,
            fontname="helv",
        )
    document.save(path)
    document.close()


def make_test_summarizer():
    from backend.app.services import passage_builder
    from backend.app.services.summarization import HierarchicalSummarizer

    with patch.object(passage_builder, "SemanticAnalyzer", FakeEmbeddingAnalyzer):
        return HierarchicalSummarizer(
            generator=FakeGenerator(),
            embedding_analyzer=FakeEmbeddingAnalyzer(),
        )


class RuntimePipelineTests(unittest.TestCase):
    def test_grounding_rejects_unsupported_identifiers_and_claims(self):
        from backend.app.services.summarization import _ground_generated_claims

        accepted, unsupported = _ground_generated_claims(
            "The Court considered Section 21. The Court imposed a fine of Rs 500 under Section 21.",
            "The Court considered Section 21. The Court imposed a fine under Section 21.",
        )

        self.assertEqual(accepted, "The Court considered Section 21.")
        self.assertEqual(len(unsupported), 1)
        self.assertIn("Identifier absent", unsupported[0]["reason"])

    def test_saved_rmu_classifier_is_loaded_for_transfer_inference(self):
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from backend.app.services import rmu_classifier

        labels = [
            "Subsumtion",
            "Vorherige Rechtsprechung des EGMR",
            "Verhältnismäßigkeitsprüfung – Angemessenheit",
            "Entscheidung des EGMR",
        ]
        features = np.eye(4, 384)
        classifier = LogisticRegression(max_iter=2000, random_state=42).fit(features, labels)

        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "classifier.joblib"
            config_path = Path(directory) / "classifier.json"
            joblib.dump({
                "classifier": classifier,
                "model_name": "all-MiniLM-L6-v2",
                "labels": labels,
            }, artifact_path)
            config_path.write_text(json.dumps({
                "model_name": "all-MiniLM-L6-v2",
                "major_labels": labels,
            }), encoding="utf-8")
            rmu_classifier._load_saved_classifier.cache_clear()
            with patch.dict(os.environ, {
                "RMU_CLASSIFIER_PATH": str(artifact_path),
                "RMU_CLASSIFIER_CONFIG": str(config_path),
            }):
                result = rmu_classifier.predict_transferred_functions(
                    [{"passage_id": 7, "embedding": features[0]}],
                    embedding_analyzer=None,
                )
            rmu_classifier._load_saved_classifier.cache_clear()

        self.assertIn(result["items"][0]["prediction"], labels)
        self.assertIn("not Indian ground truth", result["items"][0]["provenance"])

    def test_digital_pdf_extraction_preserves_pages_and_closes_document(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "digital.pdf"
            create_pdf(
                pdf_path,
                "The Supreme Court considered the appeal. "
                "The Court held that Article 21 protections applied. " * 8,
            )
            result = text_extraction.extract_pdf(pdf_path)
            pdf_path.unlink()

        self.assertEqual(result["metadata"]["page_count"], 1)
        self.assertEqual(result["pages"][0]["extraction_method"], "digital")
        self.assertEqual(result["metadata"]["pages"][0]["extraction_method"], "digital")
        self.assertTrue(result["metadata"]["quality"]["usable"])
        self.assertIn("--- Page 1 [digital] ---", result["text"])

    def test_mixed_pdf_uses_ocr_only_for_scanned_page(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "mixed.pdf"
            document = pymupdf.open()
            text_page = document.new_page()
            text_page.insert_text((50, 50), "Digital judgment page with enough readable text.")
            scan_page = document.new_page()
            image = Image.new("RGB", (900, 1200), "white")
            ImageDraw.Draw(image).text((40, 50), "Scanned judgment page", fill="black")
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")
            scan_page.insert_image(scan_page.rect, stream=image_bytes.getvalue())
            scan_page.insert_text(
                (50, 50),
                "Sparse selectable text layer from scanned page.",
            )
            document.save(pdf_path)
            document.close()

            with patch.object(text_extraction, "_ocr_page", return_value="OCR extracted judgment text") as ocr:
                result = text_extraction.extract_pdf(pdf_path)

        self.assertEqual([page["extraction_method"] for page in result["pages"]], ["digital", "ocr"])
        ocr.assert_called_once()
        self.assertIn("--- Page 2 [ocr] ---", result["text"])

    def test_hierarchical_generation_retains_references_and_relationship_metrics(self):
        text = " ".join(
            f"The Court considered appeal number {index}. The Court held Article 21 applies."
            for index in range(160)
        )
        summarizer = make_test_summarizer()
        result = summarizer.summarize(text, document_id="long-test")

        metadata = result["metadata"]
        self.assertGreater(metadata["generation_levels"], 1)
        self.assertGreater(metadata["selected_passage_count"], 0)
        self.assertTrue(metadata["generation_trace"])
        self.assertTrue(all(item["source_references"] for item in metadata["generation_trace"]))
        self.assertEqual(metadata["relationship_analysis"]["local_pair_count"],
                         sum(min(5, metadata["passage_count"] - index - 1)
                             for index in range(metadata["passage_count"])))
        self.assertFalse(metadata["functional_predictions"]["validated_on_indian_data"])
        self.assertFalse(metadata["relationship_analysis"]["actor_statistics_imported"])
        self.assertEqual(metadata["coherence_analysis"]["permutation_count"], 5)
        self.assertIn("not validated on Indian", metadata["coherence_analysis"]["status"])

    def test_small_indian_judgment_through_summarize_endpoint(self):
        source_path = ROOT / "data/experiments/gap1/Vijay_Madanlal_Choudhary_vs_Union_Of_India_on_27_July_2022_passages.json"
        source_passages = json.loads(source_path.read_text(encoding="utf-8"))
        judgment = "\n".join(passage["text"] for passage in source_passages[80:84])
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "small-judgment.pdf"
            create_pdf(pdf_path, judgment)
            pdf_bytes = pdf_path.read_bytes()
            main.UPLOAD_DIR = Path(directory) / "uploads"
            main._summarizer = make_test_summarizer()

            terminal_output = io.StringIO()
            with TestClient(main.app) as client, redirect_stdout(terminal_output):
                swagger_response = client.get("/docs")
                openapi_response = client.get("/openapi.json")
                response = client.post(
                    "/summarize",
                    files={"file": ("small-judgment.pdf", pdf_bytes, "application/pdf")},
                )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["summary"])
        self.assertIn("source_references", payload["metadata"])
        self.assertIn("grounding", payload["metadata"])
        self.assertIn("FINAL SUMMARY", terminal_output.getvalue())
        self.assertIn(payload["summary"], terminal_output.getvalue())
        self.assertEqual(swagger_response.status_code, 200)
        self.assertIn("/summarize", openapi_response.json()["paths"])
        schema = main.app.openapi()["paths"]["/summarize"]["post"]["responses"]["200"]
        self.assertEqual(
            schema["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/SummaryResponse",
        )

    def test_upload_endpoint_and_size_limit(self):
        judgment = "The Court considered the appeal and the arguments made by both parties. " * 5
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "upload.pdf"
            create_pdf(pdf_path, judgment)
            pdf_bytes = pdf_path.read_bytes()
            upload_dir = Path(directory) / "uploads"

            with patch.object(main, "UPLOAD_DIR", upload_dir), \
                    patch.object(main, "persist_text", return_value=Path(directory) / "text.txt"), \
                    TestClient(main.app) as client:
                first = client.post(
                    "/upload", files={"file": ("same-name.pdf", pdf_bytes, "application/pdf")}
                )
                second = client.post(
                    "/upload", files={"file": ("same-name.pdf", pdf_bytes, "application/pdf")}
                )
                self.assertEqual(first.status_code, 200, first.text)
                self.assertEqual(second.status_code, 200, second.text)
                stored_names = [path.name for path in upload_dir.glob("*.pdf")]
                self.assertEqual(len(stored_names), 2)
                self.assertEqual(len(set(stored_names)), 2)

            with patch.object(main, "UPLOAD_DIR", upload_dir), \
                    patch.object(main, "MAX_UPLOAD_BYTES", 8), \
                    TestClient(main.app) as client:
                oversized = client.post(
                    "/summarize",
                    files={"file": ("large.pdf", b"%PDF-1.7" + b"x" * 32, "application/pdf")},
                )
            self.assertEqual(oversized.status_code, 413)
            self.assertEqual(len(list(upload_dir.glob("*.pdf"))), 2)

    def test_vijay_madanlal_long_judgment_route_and_metadata(self):
        passages_path = ROOT / "data/experiments/gap1/Vijay_Madanlal_Choudhary_vs_Union_Of_India_on_27_July_2022_passages.json"
        passages = json.loads(passages_path.read_text(encoding="utf-8"))
        judgment = "\n".join(passage["text"] for passage in passages)

        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "vijay-madanlal.pdf"
            create_pdf(pdf_path, judgment)
            pdf_bytes = pdf_path.read_bytes()
            main.UPLOAD_DIR = Path(directory) / "uploads"
            main._summarizer = make_test_summarizer()

            terminal_output = io.StringIO()
            with TestClient(main.app) as client, redirect_stdout(terminal_output):
                response = client.post(
                    "/summarize",
                    files={"file": ("vijay-madanlal.pdf", pdf_bytes, "application/pdf")},
                )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertGreater(payload["metadata"]["passage_count"], 100)
        self.assertGreater(payload["metadata"]["selected_passage_count"], 0)
        self.assertIn("content_coverage_fraction", payload["metadata"])
        self.assertIn("functional_coverage", payload["metadata"])
        self.assertIn("coherence_signal_status", payload["metadata"]["relationship_analysis"])
        self.assertIn("FINAL SUMMARY", terminal_output.getvalue())
        self.assertIn(payload["summary"], terminal_output.getvalue())


if __name__ == "__main__":
    unittest.main()