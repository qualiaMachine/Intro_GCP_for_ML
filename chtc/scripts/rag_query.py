#!/usr/bin/env python3
"""
rag_query.py — RAG query pipeline using local open-source models.

This is the CHTC equivalent of the GCP workshop's Gemini-based RAG pipeline.
Instead of calling the Gemini API, we use:
- sentence-transformers for query embedding
- A local LLM (e.g., Gemma-2B, Mistral-7B) for generation

Usage (interactive):
    python3 rag_query.py --embeddings embeddings.npz --model google/gemma-2b-it

Usage (single query):
    python3 rag_query.py --embeddings embeddings.npz --query "What is machine learning?"
"""

import argparse
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings(path):
    """Load pre-computed embeddings and chunks."""
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["chunks"]


def retrieve(query, embeddings, chunks, model, top_k=3):
    """Find the most relevant chunks for a query."""
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "chunk": str(chunks[idx]),
            "similarity": float(similarities[idx]),
        })
    return results


def generate_answer(query, context_chunks, llm_pipeline):
    """Generate an answer using the local LLM with retrieved context."""
    context = "\n\n".join([c["chunk"] for c in context_chunks])

    prompt = f"""Based on the following context, answer the question. If the answer
is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    result = llm_pipeline(prompt, max_new_tokens=256, do_sample=True,
                          temperature=0.7, top_p=0.9)
    return result[0]["generated_text"].split("Answer:")[-1].strip()


def main():
    parser = argparse.ArgumentParser(description="RAG query with local models")
    parser.add_argument("--embeddings", required=True,
                        help="Path to embeddings.npz from embed_corpus.py")
    parser.add_argument("--query", type=str, default=None,
                        help="Single query (if omitted, enters interactive mode)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of chunks to retrieve")
    parser.add_argument("--model", type=str, default="google/gemma-2b-it",
                        help="HuggingFace model for generation")
    parser.add_argument("--embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence transformer model for query embedding")
    parser.add_argument("--retrieval_only", action="store_true",
                        help="Only retrieve chunks, skip LLM generation")
    args = parser.parse_args()

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    embeddings, chunks = load_embeddings(args.embeddings)
    print(f"Loaded {len(chunks)} chunks with {embeddings.shape[1]}-dim embeddings")

    # Load embedding model for queries
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.embedding_model)

    # Load LLM for generation (if needed)
    llm_pipeline = None
    if not args.retrieval_only:
        print(f"Loading generation model: {args.model}")
        from transformers import pipeline
        llm_pipeline = pipeline("text-generation", model=args.model,
                                device_map="auto", torch_dtype="auto")
        print("Model loaded.")

    def process_query(query):
        """Process a single query."""
        results = retrieve(query, embeddings, chunks, embed_model, args.top_k)

        print(f"\n--- Retrieved {len(results)} chunks ---")
        for i, r in enumerate(results):
            print(f"\n[Chunk {i+1}] (similarity: {r['similarity']:.4f})")
            print(r["chunk"][:200] + "..." if len(r["chunk"]) > 200 else r["chunk"])

        if llm_pipeline and not args.retrieval_only:
            print("\n--- Generated Answer ---")
            answer = generate_answer(query, results, llm_pipeline)
            print(answer)

    if args.query:
        process_query(args.query)
    else:
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if query:
                process_query(query)


if __name__ == "__main__":
    main()
