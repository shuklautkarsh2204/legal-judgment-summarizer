import spacy

nlp = spacy.blank("en") ## blank for now
nlp.add_pipe("sentencizer") ## sentence segmentation

def split_into_sentences(text: str) -> list[str]:
    doc = nlp(text)
    return [
        sent.text.strip() 
        for sent in doc.sents
        if sent.text.strip()
    ]