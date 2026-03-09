---
title: "RAG on CHTC"
teaching: 25
exercises: 15
---

::::::::::::::::::::::::::::::::::::: questions

- How can I build a RAG pipeline without cloud APIs?
- What open-source models can replace Gemini for embeddings and generation?
- How do I run embedding jobs on CHTC and LLM generation interactively?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain the RAG pattern and how it applies to CHTC workflows.
- Use sentence-transformers to generate embeddings as a batch HTCondor job.
- Run a local LLM for question answering in an interactive GPU session.

::::::::::::::::::::::::::::::::::::::::::::::::

## RAG without cloud APIs

In the [GCP version](https://qualiamachine.github.io/Intro_GCP_for_ML/) of this workshop, the RAG (Retrieval-Augmented Generation) pipeline uses Google's **Gemini API** for both embedding and generation. This requires internet access and API keys.

CHTC worker machines typically **do not have internet access**, so we use **open-source models** instead:

| RAG Component | GCP Approach | CHTC Approach |
|---|---|---|
| **Embeddings** | Gemini `text-embedding-004` API | `sentence-transformers/all-MiniLM-L6-v2` (local, no API) |
| **Generation** | Gemini `gemini-1.5-flash` API | Local LLM (Gemma-2B, Mistral-7B) via HuggingFace |
| **Execution** | Runs on Workbench notebook | Embedding as batch job; generation in interactive GPU session |
| **Internet needed?** | Yes (API calls) | No (all models run locally) |

This approach actually teaches something the GCP version doesn't — running open-source LLMs on local hardware, a valuable skill for research computing.

## The RAG pipeline

RAG has three steps, regardless of the models used:

1. **Chunk** — split your document(s) into smaller text segments.
2. **Embed** — convert each chunk into a numerical vector using an embedding model.
3. **Retrieve + Generate** — given a query, find the most relevant chunks (by vector similarity) and feed them to an LLM along with the question.

## Step 1: Embed the corpus (batch HTCondor job)

The embedding step is compute-intensive but doesn't need internet or interactivity. Perfect for a batch job.

**`embed_corpus.py`** handles:
- Extracting text from PDF (using `pypdf`)
- Splitting text into overlapping chunks
- Embedding chunks with `sentence-transformers/all-MiniLM-L6-v2`
- Saving embeddings as `.npz` (NumPy compressed)

```bash
# Test locally first (on submit node with a small document)
python3 embed_corpus.py --input document.pdf --output embeddings.npz --chunk_size 500
```

For larger corpora, submit as an HTCondor job:

```
# embed_corpus.sub

universe     = vanilla
executable   = embed_corpus.py

log          = embed_$(Cluster).log
output       = embed_$(Cluster).out
error        = embed_$(Cluster).err

request_cpus   = 4
request_memory = 4GB
request_disk   = 4GB

transfer_input_files = embed_corpus.py, document.pdf
transfer_output_files = embeddings.npz, embeddings_chunks.json

container_image = docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

arguments = --input document.pdf --output embeddings.npz

queue 1
```

::::::::::::::::::::::::::::::::::::: callout

### Why sentence-transformers?

`all-MiniLM-L6-v2` is a popular choice because:
- **Small** (~80 MB) — fast to download and run
- **CPU-friendly** — no GPU needed for embedding
- **Good quality** — strong performance on semantic similarity benchmarks
- **No API key** — runs entirely locally

For better quality (at the cost of speed), consider `all-mpnet-base-v2` (~420 MB).

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 2: Query with retrieval + generation (interactive)

The generation step is interactive — you want to see answers and iterate on queries. Use an interactive HTCondor job with a GPU:

```bash
# Request an interactive GPU session
condor_submit -i interactive_gpu.sub
```

Where `interactive_gpu.sub` requests a GPU and transfers the embedding file:

```
universe     = vanilla
executable   = /bin/bash

request_gpus   = 1
request_cpus   = 2
request_memory = 16GB
request_disk   = 20GB

transfer_input_files = rag_query.py, embeddings.npz

container_image = docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
```

Once inside the interactive session:

```bash
# Install additional dependencies (if not in container)
pip install sentence-transformers transformers accelerate

# Retrieval only (no LLM needed — just finds relevant chunks)
python3 rag_query.py --embeddings embeddings.npz --retrieval_only --query "What is machine learning?"

# Full RAG with local LLM generation
python3 rag_query.py --embeddings embeddings.npz --model google/gemma-2b-it --query "What is machine learning?"
```

::::::::::::::::::::::::::::::::::::: callout

### LLM model choices

| Model | Size | GPU Memory | Quality | Speed |
|-------|------|-----------|---------|-------|
| `google/gemma-2b-it` | 2B params | ~5 GB | Good for simple QA | Fast |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B params | ~15 GB | Better quality | Moderate |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 8B params | ~17 GB | High quality | Moderate |

For the workshop, `gemma-2b-it` is recommended — it's small enough to load quickly on any CHTC GPU and produces reasonable answers. For research use, larger models give better results.

Note: Some models (e.g., Llama) require accepting a license on HuggingFace. The model must be downloaded **before** submitting to CHTC (since workers lack internet). Download on the submit node and transfer as an input file, or use a model that's already cached.

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 3: Putting it together

The complete workflow:

```bash
# 1. Prepare your document on the submit node
cp /path/to/your/paper.pdf document.pdf

# 2. Embed as a batch job (or on submit node for small docs)
python3 embed_corpus.py --input document.pdf --output embeddings.npz

# 3. Launch interactive GPU session for querying
condor_submit -i interactive_gpu.sub

# 4. Inside the interactive session:
python3 rag_query.py --embeddings embeddings.npz --model google/gemma-2b-it
# Enters interactive mode — type questions, get answers grounded in your document
```

## Comparison with GCP approach

The GCP RAG pipeline runs entirely in a notebook with API calls. The CHTC approach separates the pipeline:

- **Embedding** → batch job (efficient, scales to large corpora)
- **Generation** → interactive session (see results immediately)

This separation is actually more flexible: you can re-embed a new corpus without re-loading the LLM, and vice versa. The embeddings are saved to disk and reused across sessions.

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Embed a document

1. Find or create a text document (or use a section of the workshop materials).
2. Run `embed_corpus.py` to create embeddings.
3. Inspect the output: how many chunks were created? What's the embedding dimension?

:::::::::::::::: solution

```bash
# Use a workshop episode as a test document
python3 embed_corpus.py --input ../episodes/01-Introduction.md --output test_embeddings.npz
```

The output will show:
- Number of chunks (depends on document length and chunk_size)
- Embedding dimension: 384 (for `all-MiniLM-L6-v2`)

Inspect with Python:
```python
import numpy as np
data = np.load("test_embeddings.npz", allow_pickle=True)
print(f"Embeddings shape: {data['embeddings'].shape}")  # (n_chunks, 384)
print(f"Number of chunks: {len(data['chunks'])}")
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Retrieval-only query

1. Using the embeddings from Challenge 1, run a retrieval-only query (no LLM needed):

```bash
python3 rag_query.py --embeddings test_embeddings.npz --retrieval_only --query "What is CHTC?"
```

2. Examine the retrieved chunks. Are they relevant to the query?
3. Try different queries and see how the similarity scores change.

:::::::::::::::: solution

The retrieval step uses cosine similarity between the query embedding and all chunk embeddings. Chunks with higher similarity scores are more semantically related to the query. Even without the LLM generation step, this retrieval mechanism is useful for:
- Finding relevant passages in large documents
- Semantic search across a corpus
- Verifying that your chunking strategy captures meaningful segments

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- CHTC workers lack internet access, so cloud APIs (Gemini) are replaced with open-source models.
- `sentence-transformers/all-MiniLM-L6-v2` provides high-quality embeddings locally without an API key.
- The embedding step runs efficiently as a batch HTCondor job; generation uses an interactive GPU session.
- This approach teaches a complementary skill: running open-source LLMs on local hardware, which is valuable beyond this workshop.

::::::::::::::::::::::::::::::::::::::::::::::::
