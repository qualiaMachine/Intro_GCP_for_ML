---
title: "Retrieval-Augmented Generation (RAG) on CHTC"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions

- How do we go from "a pile of PDFs" to "ask a question and get a cited answer"?
- What are the key parts of a RAG system (chunking, embedding, retrieval, generation)?
- How can we run a RAG pipeline on CHTC or from a submit node?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Unpack the core RAG pipeline: ingest, chunk, embed, retrieve, answer.
- Run a minimal RAG loop using foundation model APIs for embeddings and generation.
- Answer questions using content from provided papers and return grounded answers backed by source text.

::::::::::::::::::::::::::::::::::::::::::::::::

## Background concepts

This episode shifts from classical ML training (Episodes 4-6) to working with large language models (LLMs). If any of the following terms are new to you, here's a quick primer:

- **Embeddings:** A numerical vector (list of numbers) that represents the *meaning* of a piece of text. Texts with similar meanings have similar vectors. This lets us search "by meaning" rather than by keyword matching.
- **Cosine similarity:** A measure of how similar two vectors are (1.0 = identical direction, 0.0 = unrelated). Used to find which stored text chunks are most relevant to a question.
- **Large Language Model (LLM):** A model (like Gemini, GPT, or LLaMA) trained on massive text corpora that can generate coherent text given a prompt. In this episode, we use an LLM to *answer questions* based on retrieved text, not to train one from scratch.

## Overview: What we're building

**Retrieval-Augmented Generation (RAG)** is a pattern:

1. You ask a question.
2. The system **retrieves** relevant passages from your PDFs or data.
3. An LLM **answers** using those passages only, with citations.

This approach is useful any time you need to ground an LLM's answers in a specific corpus — research papers, policy documents, lab notebooks, etc.

### Running RAG on CHTC

There are two approaches for running RAG pipelines with CHTC:

1. **API-based (recommended for this workshop):** Use API-based models (Google Gemini, OpenAI) for embeddings and generation. The pipeline is lightweight enough to run interactively on the submit node for small corpora, or as an HTCondor job for larger ones.

2. **Open-source models on GPU nodes:** Run sentence-transformers for embeddings and open-source LLMs (Gemma, Mistral, LLaMA) for generation as HTCondor GPU jobs. This avoids API costs but requires more setup and larger compute resources.

We'll focus on the API approach since it's simpler and lets us focus on the RAG concepts.

### About the corpus

Our corpus is a curated bundle of **32 research papers** on the environmental and economic costs of AI — topics like training energy, inference power consumption, water footprint, and carbon emissions. They're shipped as `data/pdfs_bundle.zip` in the lesson repository. You could swap in your own PDFs — the pipeline is corpus-agnostic.

## Step 1: Set up the environment

::::::::::::::::::::::::::::::::::::: callout

### Running on the submit node vs. as a job

For a small corpus (a few dozen PDFs), the embedding and retrieval steps are lightweight enough to run directly on the submit node. For larger corpora (thousands of documents), you'd submit the embedding step as an HTCondor job with more memory and compute.

For this workshop, we'll run everything interactively on the submit node since our corpus is small.

::::::::::::::::::::::::::::::::::::::::::::::::

Install the required packages:

```bash
pip install --user pypdf scikit-learn numpy google-genai
```

::::::::::::::::::::::::::::::::::::: callout

### API key setup

Unlike the GCP version of this workshop where Vertex AI provides automatic authentication, running on CHTC requires you to set up API credentials manually.

**For Google Gemini API:**
1. Get an API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Set it as an environment variable: `export GOOGLE_API_KEY="your-key-here"`

**For OpenAI API:**
1. Get an API key from [platform.openai.com](https://platform.openai.com/api-keys).
2. Set it as an environment variable: `export OPENAI_API_KEY="your-key-here"`

**Security:** Never hardcode API keys in scripts or submit files. Use environment variables or a `.env` file that is not committed to version control.

::::::::::::::::::::::::::::::::::::::::::::::::

### Initialize project

```python
import os

# For Google Gemini API
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("Warning: GOOGLE_API_KEY not set. Set it with: export GOOGLE_API_KEY='your-key'")
```

## Step 2: Extract and chunk PDFs

Before we can search our documents, we need to break them into smaller pieces ("chunks"). Embedding models produce better vectors from focused passages than from entire papers, and LLMs have limited context windows.

```python
import zipfile, pathlib, re, pandas as pd
from pypdf import PdfReader

ZIP_PATH = pathlib.Path("Intro_GCP_for_ML/data/pdfs_bundle.zip")
DOC_DIR  = pathlib.Path("docs")
DOC_DIR.mkdir(exist_ok=True)

# unzip
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(DOC_DIR)

def chunk_text(text, max_chars=1200, overlap=150):
    for i in range(0, len(text), max_chars - overlap):
        yield text[i:i+max_chars]

rows = []
for pdf in DOC_DIR.glob("*.pdf"):
    txt = ""
    for page in PdfReader(str(pdf)).pages:
        txt += page.extract_text() or ""
    for i, chunk in enumerate(chunk_text(re.sub(r"\s+", " ", txt))):
        rows.append({"doc": pdf.name, "chunk_id": i, "text": chunk})

corpus_df = pd.DataFrame(rows)
print(len(corpus_df), "chunks created")
```

::::::::::::::::::::::::::::::::::::: callout

### Why these chunking parameters?

- **1,200 characters** (~200-300 tokens) keeps each chunk within a single focused idea.
- **150-character overlap** ensures that sentences split across chunk boundaries are still captured.
- Chunk size is a key tuning knob: smaller chunks give more precise retrieval but lose context; larger chunks preserve context but may dilute the embedding.

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 3: Embed the corpus

Now we convert each text chunk into a numerical vector so we can search by meaning rather than keywords.

```python
from google import genai
from google.genai.types import EmbedContentConfig
import numpy as np

client = genai.Client(api_key=API_KEY)

EMBED_MODEL_ID = "gemini-embedding-001"
EMBED_DIM = 1536

def embed_texts(text_list, batch_size=32, dims=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"):
    vectors = []
    for start in range(0, len(text_list), batch_size):
        batch = text_list[start : start + batch_size]
        resp = client.models.embed_content(
            model=EMBED_MODEL_ID,
            contents=batch,
            config=EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=dims,
            ),
        )
        for emb in resp.embeddings:
            vectors.append(emb.values)
    return np.array(vectors, dtype="float32")
```

### Build the retrieval index

```python
from sklearn.neighbors import NearestNeighbors

emb_matrix = embed_texts(corpus_df["text"].tolist(), dims=EMBED_DIM)
print("emb_matrix shape:", emb_matrix.shape)

nn = NearestNeighbors(metric="cosine", n_neighbors=5)
nn.fit(emb_matrix)
```

## Step 4: Retrieve and generate answers

### Retrieve relevant chunks

```python
def retrieve(query, k=5):
    query_vec = embed_texts([query], dims=EMBED_DIM, task_type="RETRIEVAL_QUERY")[0]
    distances, indices = nn.kneighbors([query_vec], n_neighbors=k, return_distance=True)
    result_df = corpus_df.iloc[indices[0]].copy()
    result_df["similarity"] = 1 - distances[0]
    return result_df.sort_values("similarity", ascending=False)
```

### Generate a grounded answer

```python
from google.genai.types import GenerateContentConfig

GENERATION_MODEL_ID = "gemini-2.5-flash"

def ask(query, top_k=5, temperature=0.2):
    hits = retrieve(query, k=top_k)
    context_lines = [
        f"[{row.doc}#chunk-{row.chunk_id}] {row.text}"
        for _, row in hits.iterrows()
    ]
    context_block = "\n\n".join(context_lines)

    prompt = (
        "You are a research assistant. "
        "Use only the following context to answer the question. "
        "Cite your sources using the [doc#chunk] tags.\n\n"
        f"{context_block}\n\n"
        f"Q: {query}\n"
        "A:"
    )

    response = client.models.generate_content(
        model=GENERATION_MODEL_ID,
        contents=prompt,
        config=GenerateContentConfig(temperature=temperature),
    )
    return response.text
```

### Test the pipeline

```python
print(ask("How much energy does it cost to train a large language model?"))
```

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 1: Explore chunk size tradeoffs

Change the `max_chars` parameter in `chunk_text()` to **500** and then to **2500**. Re-run the chunking, embedding, and retrieval steps each time, then ask the same question.

- How does the number of chunks change?
- Does the answer quality improve or degrade?
- Which chunk size gives the best balance of precision and context?

:::::::::::::::::::::::: solution

Smaller chunks (500 chars) produce more precise retrieval hits but each chunk has less context, so the LLM may struggle to synthesize a complete answer. Larger chunks (2,500 chars) preserve more context but may dilute the embedding with unrelated text. For most research-paper corpora, 800-1,500 characters is a practical sweet spot.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 2: Test hallucination behavior

Ask a question that has **no answer** in the corpus:

```python
print(ask("What was the GDP of France in 2019?"))
```

- Does the LLM refuse to answer, or does it hallucinate?
- Try modifying the system prompt in `ask()` to add: *"If the context does not contain enough information to answer, say 'I don't have enough information to answer this.'"*
- Does the modified prompt change the behavior?

:::::::::::::::::::::::: solution

Without the guardrail prompt, the LLM may produce a plausible-sounding answer from its training data. Adding an explicit refusal instruction significantly reduces hallucination. **Prompt engineering is part of RAG system design**, not just model selection.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 3: Tune retrieval depth with `top_k`

Call `ask()` with `top_k=2` and then with `top_k=10`. Compare the answers.

- With `top_k=2`, does the LLM miss relevant information?
- With `top_k=10`, does the extra context help or introduce noise?
- What value of `top_k` seems to work best for your question?

:::::::::::::::::::::::: solution

Lower `top_k` gives a tighter, more focused context — good when the answer is localized in one or two chunks. Higher `top_k` provides broader coverage but risks including irrelevant passages. A good default is 3-5 for most research-paper RAG tasks.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Challenge 4: Try different questions

```python
# Off-topic question
print(ask("How much does an elephant weigh?"))

# Comparative question — requires synthesizing across sources
print(ask("Is cloud computing more energy efficient than university HPC clusters?"))

# Opinion question — may tempt the model to go beyond the corpus
print(ask("What is the most energy-efficient way to train a neural network?"))
```

For each question, consider:
- Does the answer cite specific numbers or papers from the corpus?
- Does the LLM stay grounded in the retrieved context?
- Which question produces the most useful, well-supported answer?

:::::::::::::::::::::::: solution

The elephant-weight question is deliberately off-topic — a well-behaved RAG system should indicate the context doesn't contain relevant information. The comparative and opinion questions require synthesis across sources — look for whether the model hedges appropriately when papers disagree.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Running RAG as an HTCondor job

For larger corpora, you can submit the embedding step as an HTCondor job. Create a script that reads PDFs, computes embeddings, and saves them to a file:

```bash
# rag_embed.sub
universe = docker
docker_image = python:3.10-slim

executable = run_rag_embed.sh
transfer_input_files = rag_embed.py, pdfs_bundle.zip, requirements_rag.txt

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

log    = logs/rag_embed_$(Cluster).log
output = logs/rag_embed_$(Cluster).out
error  = logs/rag_embed_$(Cluster).err

request_cpus   = 2
request_memory = 8GB
request_disk   = 4GB

# Pass API key as an environment variable
environment = "GOOGLE_API_KEY=$(GOOGLE_API_KEY)"

queue 1
```

Then do retrieval and generation interactively with the pre-computed embeddings.

::::::::::::::::::::::::::::::::::::: callout

### Open-source alternatives

You can replace the API-based models with open-source alternatives that run entirely on CHTC GPU nodes:

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-large-en-v1.5`
- **Generators:** `google/gemma-2b-it`, `mistralai/Mistral-7B-Instruct`

This requires GPU jobs with appropriate Docker images (e.g., a container with PyTorch and transformers installed). The advantage is no API costs and no external dependencies; the tradeoff is more setup and longer queue times for GPU resources.

::::::::::::::::::::::::::::::::::::::::::::::::

### Cleanup note

The embeddings and nearest-neighbors index in this episode are held **in memory** — they disappear when your Python session ends. No persistent CHTC resources were created beyond the files in your home directory. Clean up any large temporary files (extracted PDFs, cached embeddings) when you're done.

::::::::::::::::::::::::::::::::::::: keypoints

- RAG grounds LLM answers in your own data — retrieve first, then generate.
- The pipeline (chunk, embed, retrieve, generate) works the same regardless of where you run it.
- Chunk size, retrieval depth (`top_k`), and prompt design are the primary tuning levers.
- API-based models are simplest for small corpora; open-source models on CHTC GPUs avoid API costs for larger workloads.
- Always cite retrieved chunks for reproducibility and transparency.

::::::::::::::::::::::::::::::::::::::::::::::::
