import re
from pathlib import Path
from preprocessing import split_into_sentences

def analyze_document(pdf_path: Path)->dict:
    text = pdf_path.read_text(encoding="utf-8")
    
    words = re.findall(r'\w+',text)
    word_count = len(words)
    
    sentences = split_into_sentences(text)
    sentence_count = len(
        [sentence for sentence in sentences if sentence.strip()]
    )
    
    estimated_tokens = int(word_count * 1.3)  ## rough estimate of tokens based on word count
    
    characters = len(text)
    return{
        "characters": characters,
        "words": word_count,
        "sentences": sentence_count,
        "estimated_tokens": estimated_tokens
    }