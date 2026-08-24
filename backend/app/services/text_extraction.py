import pymupdf as pmpdf
from pathlib import Path

def extract_text_from_pdf(pdf_path: Path) -> str:
    document = pmpdf.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text()
    
    document.close()
    return text    

def persist_text(text: str, pdf_path: Path) -> Path:
    processed_dir = Path("data/processed") ## location of extraced text.
    processed_dir.mkdir(parents=True, exist_ok=True)
    text_filename = pdf_path.stem + ".txt"
    text_path = processed_dir / text_filename
    
    with open(text_path, "w" , encoding="utf-8") as file:
        file.write(text)
    
    return text_path    