from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from backend.app.services.text_extraction import extract_text_from_pdf, persist_text

app = FastAPI(
    title = "Indian Legal Judgemenr Summarizer",
    description = "NLP-based legal judgement summarization system"
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def root():
    return {
        "project": "Indian Legal Judgemenr Summarizer",
        "status": "running"
    }

@app.post("/upload") ## endpoint to upload
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only PDF files are allowed."
        )  
    file_path = UPLOAD_DIR / file.filename ## uploaded file are stored here.
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    
    text = extract_text_from_pdf(file_path)  ## extract text from pdf  
    text_path = persist_text(text, file_path) ## persist the extracted text to a file.  
    
    return {
        "filename": file.filename,
        "size_bytes": len(contents),
        "characters_extracted": len(text),
        "status": "uploaded and processed",
        "text_file": str(text_path),
        "text_preview": text[:200] ## preview the first 200 characters.
    }          