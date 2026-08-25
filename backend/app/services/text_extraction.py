import pymupdf as pmpdf
from pathlib import Path

def extract_text_from_pdf(pdf_path: Path) -> str:
    document = pmpdf.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text()
    
    document.close()
    return text 

def is_txt_enough(text: str) -> bool:
    if len(text.strip()) >= 100: ## threshold is 100 words.
        return True   
    return False

def txt_extraction(pdf_path: Path)->str:
    try:
        text = extract_text_from_pdf(pdf_path)
        
        if is_txt_enough(text):
            return text
        else:
            raise ValueError("Extracted text is not enough.")
    except ValueError as e:
        raise NotImplementedError(
            "PDF is scanned and OCR is not yet implemented."
        )    
        
    

def persist_text(text: str, pdf_path: Path) -> Path:
    processed_dir = Path("data/processed") ## location of extraced text.
    processed_dir.mkdir(parents=True, exist_ok=True)
    text_filename = pdf_path.stem + ".txt"
    text_path = processed_dir / text_filename
    
    with open(text_path, "w" , encoding="utf-8") as file:
        file.write(text)
    
    return text_path    