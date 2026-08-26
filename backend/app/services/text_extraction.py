import io
import pymupdf as pmpdf
from pathlib import Path
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  ## configuring tesseract

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
    
    text = extract_text_from_pdf(pdf_path)

    if is_txt_enough(text):
        return text
    else:
        return extract_with_OCR(pdf_path)  ## if text is not enough, ofc length would be zero use ocr...
       
        
def extract_with_OCR(pdf_path:Path)->str:
    document = pmpdf.open(pdf_path)
    extracted_pages = []
    for page, page_number in enumerate(document, start=1):
        pix = page.get_pixmap(dpi=300) ## rendering each page of pdf as img.
        img_bytes = pix.tobytes("png")
        img = Image.open(
            io.BytesIO(img_bytes)
        )
        page_text = pytesseract.image_to_string(img, lang = "eng") 
        extracted_pages.append(
            f"\n--- Page {page_number} --- \n{page_text}"
        )
        document.close()
    return "\n".join(extracted_pages)   

def persist_text(text: str, pdf_path: Path) -> Path:
    processed_dir = Path("data/processed") ## location of extraced text.
    processed_dir.mkdir(parents=True, exist_ok=True)
    text_filename = pdf_path.stem + ".txt"
    text_path = processed_dir / text_filename
    
    with open(text_path, "w" , encoding="utf-8") as file:
        file.write(text)
    
    return text_path    