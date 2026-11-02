"""PDF text extraction with page-level OCR fallback and provenance metadata."""

import io
import importlib
import os
import re
from pathlib import Path


def _pdf_dependencies():
    try:
        return (
            importlib.import_module("pymupdf"),
            importlib.import_module("pytesseract"),
            importlib.import_module("PIL.Image"),
        )
    except ImportError as error:
        raise RuntimeError(
            "PDF extraction requires PyMuPDF, pytesseract, and Pillow."
        ) from error


def _configure_tesseract(pytesseract):
    configured_path = os.getenv("TESSERACT_CMD")
    if configured_path:
        pytesseract.pytesseract.tesseract_cmd = configured_path


def _needs_ocr(page, text):
    alphabetic_characters = sum(character.isalpha() for character in text)
    try:
        image_info = page.get_image_info(xrefs=True)
    except AttributeError:
        image_info = page.get_images(full=True)

    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    image_area = 0.0
    for image in image_info:
        bbox = image.get("bbox") if isinstance(image, dict) else None
        if bbox:
            image_area += max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    has_large_image = image_area / page_area >= 0.2
    if has_large_image and alphabetic_characters < 150:
        return True
    if alphabetic_characters >= 40:
        return False
    return has_large_image or (not text.strip() and bool(image_info))


def _ocr_page(page, pytesseract, image_module, dpi=300, language="eng"):
    _configure_tesseract(pytesseract)
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    image = image_module.open(io.BytesIO(pixmap.tobytes("png")))
    return pytesseract.image_to_string(image, lang=language).strip()


def assess_extraction_quality(text, pages):
    words = re.findall(r"\b[\w'-]+\b", text)
    alphabetic_characters = sum(character.isalpha() for character in text)
    pages_with_text = sum(bool(page["text"].strip()) for page in pages)
    warnings = []

    if alphabetic_characters < 100:
        warnings.append("Very little alphabetic text was extracted.")
    if pages and pages_with_text / len(pages) < 0.5:
        warnings.append("Fewer than half of the pages contain extractable text.")

    return {
        "word_count": len(words),
        "alphabetic_characters": alphabetic_characters,
        "page_count": len(pages),
        "pages_with_text": pages_with_text,
        "usable": alphabetic_characters >= 30,
        "warnings": warnings,
    }


def extract_pdf(pdf_path: Path, *, ocr_dpi=None, ocr_language=None) -> dict:
    """Extract each page digitally or with OCR, preserving page provenance."""
    pdf_path = Path(pdf_path)
    pmpdf, pytesseract, image_module = _pdf_dependencies()
    dpi = int(ocr_dpi or os.getenv("OCR_DPI", "300"))
    language = ocr_language or os.getenv("TESSERACT_LANG", "eng")
    pages = []

    with pmpdf.open(pdf_path) as document:
        page_count = len(document)
        for page_number, page in enumerate(document, start=1):
            digital_text = page.get_text("text").strip()
            extraction_method = "digital"
            page_text = digital_text

            if _needs_ocr(page, digital_text):
                try:
                    page_text = _ocr_page(
                        page, pytesseract, image_module, dpi=dpi, language=language
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"OCR failed on PDF page {page_number}: {error}"
                    ) from error
                extraction_method = "ocr"

            pages.append({
                "page_number": page_number,
                "extraction_method": extraction_method,
                "character_count": len(page_text),
                "text": page_text,
            })

    text = "\n".join(
        f"--- Page {page['page_number']} [{page['extraction_method']}] ---\n{page['text']}"
        for page in pages
    )
    methods = sorted({page["extraction_method"] for page in pages})
    page_metadata = [
        {
            "page_number": page["page_number"],
            "extraction_method": page["extraction_method"],
            "character_count": page["character_count"],
        }
        for page in pages
    ]
    return {
        "text": text,
        "pages": pages,
        "metadata": {
            "source_filename": pdf_path.name,
            "page_count": page_count,
            "pages": page_metadata,
            "extraction_methods": methods,
            "quality": assess_extraction_quality(text, pages),
        },
    }


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract embedded digital text only; retained for compatibility."""
    pmpdf, _, _ = _pdf_dependencies()
    with pmpdf.open(pdf_path) as document:
        return "\n".join(page.get_text("text") for page in document)


def is_txt_enough(text: str) -> bool:
    return sum(character.isalpha() for character in text) >= 30


def txt_extraction(pdf_path: Path) -> str:
    return extract_pdf(pdf_path)["text"]


def extract_with_OCR(pdf_path: Path) -> str:
    """OCR every page; retained for compatibility with the earlier helper."""
    pages = []
    pmpdf, pytesseract, image_module = _pdf_dependencies()
    with pmpdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = _ocr_page(page, pytesseract, image_module)
            pages.append(f"\n--- Page {page_number} [ocr] ---\n{page_text}")
    return "\n".join(pages)


def persist_text(text: str, pdf_path: Path) -> Path:
    processed_dir = Path(os.getenv("PROCESSED_DIR", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    text_path = processed_dir / f"{Path(pdf_path).stem}.txt"
    text_path.write_text(text, encoding="utf-8")
    return text_path