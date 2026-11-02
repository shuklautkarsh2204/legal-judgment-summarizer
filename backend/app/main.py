"""FastAPI application for the Indian legal judgment summarizer."""

import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.app.services.text_extraction import extract_pdf, persist_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", PROJECT_ROOT / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
CHUNK_SIZE = 1024 * 1024
logger = logging.getLogger("legal_judgment_summarizer")
_summarizer = None


class SummaryResponse(BaseModel):
    summary: str
    metadata: dict[str, Any]


app = FastAPI(
    title="Indian Legal Judgment Summarizer",
    description=(
        "Hierarchical, source-referenced judgment summaries. Experimental transfer "
        "signals are identified separately and are not Indian ground truth."
    ),
    version="1.0.0",
)


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        from backend.app.services.summarization import HierarchicalSummarizer

        _summarizer = HierarchicalSummarizer()
    return _summarizer


async def _store_upload(file: UploadFile) -> tuple[Path, int, str]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / f"{uuid4().hex}.pdf"
    byte_count = 0
    signature = bytearray()

    try:
        with file_path.open("wb") as destination:
            while chunk := await file.read(CHUNK_SIZE):
                byte_count += len(chunk)
                if byte_count > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF exceeds the {MAX_UPLOAD_BYTES}-byte upload limit.",
                    )
                if len(signature) < 5:
                    signature.extend(chunk[:5 - len(signature)])
                destination.write(chunk)

        if byte_count == 0 or bytes(signature) != b"%PDF-":
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")
        return file_path, byte_count, file.filename or "uploaded.pdf"
    except Exception:
        file_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


def _extract(file_path, original_filename):
    logger.info("Extracting PDF text from %s", original_filename)
    extraction = extract_pdf(file_path)
    extraction["metadata"]["source_filename"] = original_filename
    quality = extraction["metadata"]["quality"]
    if not quality["usable"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "PDF extraction produced too little usable text.",
                "extraction_quality": quality,
            },
        )
    return extraction


@app.get("/")
def root():
    return {"project": "Indian Legal Judgment Summarizer", "status": "running"}


@app.post("/upload", summary="Extract text from a judgment PDF")
async def upload_file(file: UploadFile = File(...)):
    file_path = None
    try:
        file_path, byte_count, original_filename = await _store_upload(file)
        extraction = _extract(file_path, original_filename)
        text_path = persist_text(extraction["text"], file_path)
        return {
            "filename": original_filename,
            "size_bytes": byte_count,
            "characters_extracted": len(extraction["text"]),
            "status": "uploaded and processed",
            "text_file": str(text_path),
            "text_preview": extraction["text"][:200],
            "metadata": extraction["metadata"],
        }
    except HTTPException:
        raise
    except (RuntimeError, ImportError) as error:
        logger.exception("PDF extraction dependency or OCR failure")
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("PDF upload processing failed")
        raise HTTPException(status_code=422, detail=f"Unable to process PDF: {error}") from error


@app.post(
    "/summarize",
    summary="Summarize a judgment PDF",
    response_model=SummaryResponse,
)
async def summarize_file(file: UploadFile = File(...)) -> SummaryResponse:
    file_path = None
    try:
        file_path, byte_count, original_filename = await _store_upload(file)
        extraction = _extract(file_path, original_filename)
        logger.info("Starting hierarchical analysis for %s", original_filename)
        result = get_summarizer().summarize(
            extraction["text"],
            document_id=Path(original_filename).stem,
            extraction_metadata=extraction["metadata"],
        )
        result["metadata"]["filename"] = original_filename
        result["metadata"]["upload_size_bytes"] = byte_count
        print(f"FINAL SUMMARY [{original_filename}]\n{result['summary']}", flush=True)
        logger.info("Final summary for %s:\n%s", original_filename, result["summary"])
        return result
    except HTTPException:
        raise
    except (RuntimeError, ImportError) as error:
        logger.exception("Summarization model, dependency, or OCR failure")
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        logger.exception("Summarization input validation failed")
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Summarization failed")
        raise HTTPException(
            status_code=500,
            detail="Summarization failed. Check server logs for the underlying error.",
        ) from error
    finally:
        if file_path is not None:
            file_path.unlink(missing_ok=True)