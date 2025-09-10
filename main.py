import torch
import fitz
import re
from keybert import KeyBERT
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


pipe = pipeline("summarization", model="facebook/bart-large-cnn")


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text+=page.get_text()
    return text

def clean_pdf_text(text: str) -> str:
    # normalize bullets and similar tokens
    text = re.sub(r"[•●▪◦]+", " ", text)
    # remove isolated numbers or section counters like "1." or "2 "
    text = re.sub(r"\b\d+\s*\.\s*", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    # convert single newlines to spaces, keep paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # trim leading/trailing whitespace
    return text.strip()

def chunk_text(text, max_tokens=512):
    tokens = tokenizer.encode(text)
    return [
        tokenizer.decode(tokens[i:i+max_tokens], skip_special_tokens=True)
        for i in range(0, len(tokens), max_tokens)
    ]


def extract_key_clauses(text: str, top_n: int = 5):
    # Initialize model (MiniLM is small and fast)
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    # Extract candidate clauses (keywords or key phrases)
    key_phrases = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(3, 10),  # look for clauses not just single words
        stop_words='english',
        top_n=top_n,
        use_maxsum=True
    )
    return [phrase for phrase, score in key_phrases]



def run_pipeline(file_path):
    text = extract_text_from_pdf(file_path)
    text = clean_pdf_text(text)
    chunks = chunk_text(text)
    summaries = [pipe(c, max_length=130, min_length=30, truncation=True)[0]["summary_text"]
             for c in chunks]
    key_clauses = []
    for chunk in chunks:
        key_clauses.extend(extract_key_clauses(chunk, top_n=4))
    return {
        "summary": summaries,
        "clauses": key_clauses
    }

if __name__ == "__main__":
    import sys, json
    pdf = sys.argv[1]
    print(json.dumps(run_pipeline(pdf)))

