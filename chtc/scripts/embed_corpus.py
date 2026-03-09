#!/usr/bin/env python3
"""
embed_corpus.py — Embed PDF text chunks using sentence-transformers.

This is the CHTC equivalent of the GCP workshop's Gemini embedding API calls.
Instead of calling a cloud API, we run a local open-source model
(sentence-transformers/all-MiniLM-L6-v2) that works without internet access.

Usage:
    python3 embed_corpus.py --input document.pdf --output embeddings.npz
    python3 embed_corpus.py --input chunks.txt --output embeddings.npz
"""

import argparse
import json
import numpy as np
from pathlib import Path


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Embed text chunks using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Embed document chunks for RAG")
    parser.add_argument("--input", required=True,
                        help="Input file (PDF or plain text)")
    parser.add_argument("--output", default="embeddings.npz",
                        help="Output .npz file with embeddings and chunks")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Number of words per chunk")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Word overlap between chunks")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence transformer model name")
    args = parser.parse_args()

    # Load and chunk the document
    input_path = Path(args.input)
    if input_path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(args.input)
    else:
        text = input_path.read_text()

    chunks = chunk_text(text, args.chunk_size, args.overlap)
    print(f"Created {len(chunks)} chunks from {args.input}")

    # Embed
    embeddings = embed_chunks(chunks, args.model)

    # Save embeddings and chunks together
    np.savez(args.output,
             embeddings=embeddings,
             chunks=np.array(chunks, dtype=object))

    # Also save chunks as JSON for easy inspection
    chunks_path = args.output.replace(".npz", "_chunks.json")
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved embeddings: {args.output} (shape: {embeddings.shape})")
    print(f"Saved chunks: {chunks_path}")


if __name__ == "__main__":
    main()
