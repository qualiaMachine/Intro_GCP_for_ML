---
title: "Retrieval-Augmented Generation (RAG) with Vertex AI"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions

- How do we go from "a pile of PDFs" to "ask a question and get a cited answer" using Google Cloud tools?
- What are the key parts of a RAG system (chunking, embedding, retrieval, generation), and how do they map onto Vertex AI services?
- How much does each part of this pipeline cost (VM time, embeddings, LLM calls), and where can we keep it cheap?
- Can we use open models / Hugging Face instead of Google models, and what does that change?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Unpack the core RAG pipeline: ingest → chunk → embed → retrieve → answer.
- Run a minimal, fully programmatic RAG loop on a Vertex AI Workbench VM using Google’s own foundation models (for embeddings + generation).
- Understand how to substitute open-source / Hugging Face models if you want to avoid managed API costs.
- Answer questions using content from provided papers and return citations instead of vibes.

::::::::::::::::::::::::::::::::::::::::::::::::

## Overview: What we're building

**Retrieval-Augmented Generation (RAG)** is a pattern:

1. You ask a question.  
2. The system **retrieves** relevant passages from your PDFs or data.  
3. An LLM **answers** using those passages only, with citations.

This approach powers sustainability-related projects like **WattBot**, which extracts AI water and energy metrics from research papers.

**Cost mindset:**  
- **VM cost:** pay for Workbench instance uptime. Stop when not in use.  
- **Embedding cost:** pay per character embedded — only once per doc.  
- **Generation cost:** pay per token for input + output. Shorter prompts = cheaper.  

**Hugging Face alternatives:**  
You can replace Google-managed APIs with open models such as:  
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-large-en-v1.5`  
- **Generators:** `google/gemma-2b-it`, `mistralai/Mistral-7B-Instruct`, or `tiiuae/falcon-7b-instruct`  
However, this requires a GPU or large CPU VM (e.g., `n1-standard-8` + `T4`) and manual model management. Rather than use a very expensive machine and GPU in Workbench, you can launch custom jobs that perform the embedding and generation steps. Start with a PyTorch image and add HuggingFace as a requirement.



## Step 1: Setup environment

We need the `pypdf` library to extract text from PDF files. The `--quiet` flag suppresses verbose install output, and `--upgrade` ensures we have the latest version.

```python
!pip install --quiet --upgrade pypdf
```

**Cost note:** Installing packages is free; you're only billed for VM runtime.

### Initialize project

We initialize two client libraries here because they serve different purposes. The `aiplatform` SDK (used in earlier episodes for training jobs) handles resource management, while the `vertexai` SDK provides access to Google's foundation models (embeddings and Gemini). Both need to know your project ID and region.

```python
from google.cloud import aiplatform
from vertexai import init as vertexai_init
import os

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "<YOUR_PROJECT_ID>")
REGION = "us-central1"

aiplatform.init(project=PROJECT_ID, location=REGION)
vertexai_init(project=PROJECT_ID, location=REGION)
print("Initialized:", PROJECT_ID, REGION)
```



## Step 2: Extract and chunk PDFs

Before we can search our documents, we need to break them into smaller pieces ("chunks"). LLMs have limited context windows and embedding models work best on passages rather than entire papers. The code below extracts text from each PDF and splits it into overlapping chunks of ~1,200 characters. The overlap ensures that sentences at chunk boundaries aren't lost.

```python
import zipfile, pathlib, re, pandas as pd
from pypdf import PdfReader

ZIP_PATH = pathlib.Path("/home/jupyter/Intro_GCP_for_ML/data/pdfs_bundle.zip")
DOC_DIR = pathlib.Path("/home/jupyter/docs")
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

**Cost note:** Only VM runtime applies. Chunk size affects future embedding cost.



## Step 3: Embed text using Vertex AI

### Choosing an embedding and generator model

Vertex AI currently offers multiple managed embedding models under the **Text Embeddings API** family.  
For this exercise, we’re using **`text-embedding-004`**, which is Google’s latest general-purpose model optimized for **semantic similarity**, **retrieval**, and **clustering** tasks.  

**Why this model?**
- Produces 768-dimensional dense vectors suitable for cosine or dot-product similarity.  
- Handles long passages (up to ~8,000 tokens) and multilingual content.  
- Tuned for retrieval tasks like RAG, document search, and clustering.  
- Cost-efficient for classroom-scale workloads (fractions of a cent per document).  

If you’d like to explore other options:
- Open the [**Vertex AI Model Garden → Text Embeddings**](https://console.cloud.google.com/vertex-ai/model-garden?project=doit-rci-mlm25-4626&pageState=(%22galleryStateKey%22:(%22f%22:(%22g%22:%5B%22goals%22%5D,%22o%22:%5B%22Text%20embeddings%22%5D),%22s%22:%22%22))) in your GCP console. <!-- replace project ID with your own if not using the shared workshop project -->  
- You’ll find specialized alternatives such as:
  - **`text-embedding-005` (experimental)** – larger model, higher precision on longer documents.  
  - **`multimodal-embedding-001`** – supports image + text embeddings for richer use cases.  
  - **Third-party embeddings (via Model Garden)** – e.g., `bge-large-en`, `cohere-embed-v3`, `all-MiniLM`.  



The following code sets up the Gen AI client and chooses our embedding and generation models. We use `vertexai=True` to route API calls through your GCP project (for billing and governance) rather than the public endpoint.

```python
#############################################
# 1. Imports and client setup
#############################################

from google import genai
from google.genai.types import HttpOptions, EmbedContentConfig, GenerateContentConfig
import numpy as np
from sklearn.neighbors import NearestNeighbors

# We'll assume you already have:
#   corpus_df  -> pandas DataFrame with columns: 'text', 'doc', 'chunk_id'
# If not, you'll need to define/load that before running this cell.


#############################################
# 2. Initialize the Gen AI client
#############################################

# vertexai=True = bill/govern in your GCP project instead of the public endpoint
client = genai.Client(
    http_options=HttpOptions(api_version="v1"),
    vertexai=True,
    project=PROJECT_ID,
    location="us-central1",
)

# Generation model for answering questions
GENERATION_MODEL_ID = "gemini-2.5-pro"        # or "gemini-2.5-flash" for cheaper/faster

# Embedding model for retrieval
EMBED_MODEL_ID = "gemini-embedding-001"

# Pick an embedding dimensionality and stick to it across corpus + queries.
EMBED_DIM = 1536  # valid typical choices: 768, 1536, 3072


```

Next, we define a helper function that converts text strings into numerical vectors (embeddings). It processes texts in batches to avoid hitting API request-size limits. The `RETRIEVAL_DOCUMENT` task type tells the model to optimize embeddings for search/retrieval use cases.

```python
#############################################
# 3. Helper: get embeddings for a list of texts
#############################################

def embed_texts(text_list, batch_size=32, dims=EMBED_DIM):
    """
    Convert a list of text strings into embedding vectors using gemini-embedding-001.
    Returns a NumPy array of shape (len(text_list), dims).
    """
    vectors = []

    # batch to avoid huge single requests
    for start in range(0, len(text_list), batch_size):
        batch = text_list[start:start+batch_size]

        resp = client.models.embed_content(
            model=EMBED_MODEL_ID,
            contents=batch,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",   # optimize embeddings for retrieval/use as chunks
                output_dimensionality=dims,       # must match EMBED_DIM everywhere
            ),
        )

        # resp.embeddings is aligned with 'batch'
        for emb in resp.embeddings:
            vectors.append(emb.values)

    return np.array(vectors, dtype="float32")


```

Now we embed all the corpus chunks and build a nearest-neighbors index for fast retrieval. The `NearestNeighbors` model from scikit-learn uses cosine similarity to find the chunks most relevant to a query. We also define a `retrieve()` function that embeds a user question and returns the top-k matching chunks.

```python
#############################################
# 4. Embed the corpus and build the NN index
#############################################

# Create embeddings for every text chunk in the corpus
emb_matrix = embed_texts(corpus_df["text"].tolist(), dims=EMBED_DIM)
print("emb_matrix shape:", emb_matrix.shape)   # (num_chunks, EMBED_DIM)

# Fit NearestNeighbors on those embeddings once
nn = NearestNeighbors(
    metric="cosine",   # cosine distance is standard for semantic similarity
    n_neighbors=5,     # default neighborhood size; can override at query time
)
nn.fit(emb_matrix)


#############################################
# 5. Retrieval: given a query string, get top-k relevant chunks
#############################################

def retrieve(query, k=5):
    """
    Embed the user query with the SAME embedding model/dim,
    then find the top-k most similar corpus chunks.
    Returns a DataFrame of the top matches with a 'similarity' column.
    """

    # Embed the query to the same dimension space as emb_matrix
    query_vec = embed_texts([query], dims=EMBED_DIM)[0]   # shape (EMBED_DIM,)

    # Find nearest neighbors using cosine distance
    distances, indices = nn.kneighbors([query_vec], n_neighbors=k, return_distance=True)

    # Grab those rows from the original corpus
    result_df = corpus_df.iloc[indices[0]].copy()

    # Convert cosine distance -> cosine similarity (1 - distance)
    result_df["similarity"] = 1 - distances[0]

    # Sort by similarity descending (highest similarity first)
    result_df = result_df.sort_values("similarity", ascending=False)

    return result_df

```

Finally, the `ask()` function ties the full RAG pipeline together: it retrieves relevant chunks, constructs a prompt that includes only those chunks as context, and sends the prompt to Gemini for a grounded answer. The `temperature` parameter controls how creative the model's response is — lower values produce more factual, deterministic answers.

```python
#############################################
# 6. ask(): build grounded prompt + call Gemini to answer
#############################################

def ask(query, top_k=5, temperature=0.2):
    """
    Retrieval-Augmented Generation:
    - retrieve context chunks relevant to `query`
    - stuff those chunks into a prompt
    - ask Gemini to answer ONLY using that context
    """

    # Get top_k most relevant text chunks
    hits = retrieve(query, k=top_k)

    # Build a context block with provenance tags like [doc#chunk-id]
    context_lines = [
        f"[{row.doc}#chunk-{row.chunk_id}] {row.text}"
        for _, row in hits.iterrows()
    ]
    context_block = "\n\n".join(context_lines)

    # Instruction prompt for the model
    prompt = (
        "You are a sustainability analyst. "
        "Use only the following context to answer the question.\n\n"
        f"{context_block}\n\n"
        f"Q: {query}\n"
        "A:"
    )

    # Call the generative model
    response = client.models.generate_content(
        model=GENERATION_MODEL_ID,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=temperature,  # lower = more deterministic, factual
        ),
    )

    # Return the model's answer text
    return response.text

```

## Step 4: Generate answers using Gemini

With all the pieces in place, let's test the full pipeline end-to-end. The `ask()` function retrieves relevant chunks from our corpus, passes them as context to Gemini, and returns a grounded answer with citations.

```python

#############################################
# 7. Test the pipeline end-to-end
#############################################

print(
    ask(
        "What is the name of the benchmark suite presented in a recent paper "
        "for measuring inference energy consumption?"
    )
)
# Expected answer: "ML.ENERGY Benchmark"

```


## Step 5: Cost summary

Understanding the cost of each pipeline component helps you decide where to optimize. The table below breaks down what you pay for at each stage. For a small workshop with a handful of PDFs, total costs are typically well under $1.

| Step | Resource | Example Component | Cost Driver | Typical Range |
|------|-----------|-------------------|--------------|----------------|
| VM runtime | Vertex AI Workbench | `n1-standard-4` | Uptime (hourly) | ~$0.20/hr |
| Embeddings | text-embedding-004 | Managed API | Tokens embedded | ~$0.10 / 1M tokens |
| Retrieval | Local NN | CPU only | None | Free |
| Generation | gemini-2.5-flash-001 | Managed API | Input/output tokens | ~$0.25 / 1M tokens |
| Hugging Face alt | T4 VM | Local model inference | GPU uptime | ~$0.35/hr + egress |


## (Optional) Hugging Face local substitution

To avoid managed API costs, you can use Hugging Face models instead. This section is left as an exercise — refer to the Hugging Face alternatives listed at the top of this episode for model suggestions.

## Key takeaways

- Use **Vertex AI managed embeddings** and **Gemini Flash** for lightweight, cost-controlled RAG.
- Cache embeddings; reusing them saves most cost.
- For open alternatives, use Hugging Face models on GPU VMs (higher cost, more control).
- This workflow generalizes to any retrieval task — not just sustainability papers.
- GCP’s managed tools lower barrier for experimentation while keeping enterprise security and IAM intact.

## What’s next?

This workshop covered the core GCP + Vertex AI workflow: data storage, notebook-driven job submission, training, hyperparameter tuning, and RAG. Here are natural next steps depending on your goals:

- **Vertex AI Pipelines** — Automate multi-step ML workflows (preprocessing → training → evaluation → deployment) as reproducible DAGs using Kubeflow Pipelines or TFX on Vertex AI.
- **Model Registry & Endpoints** — Register your trained models in Vertex AI’s Model Registry and deploy them as online prediction endpoints for serving.
- **Vertex AI Experiments** — Track and compare runs systematically, beyond what `metrics.json` files in GCS provide.
- **BigQuery ML** — If your data lives in BigQuery, you can train and predict directly in SQL without moving data to GCS.
- **Cost optimization** — Explore [committed use discounts](https://cloud.google.com/compute/docs/instances/committed-use-discounts-overview) and [preemptible/spot VMs](https://cloud.google.com/compute/docs/instances/spot) for significant savings on long-running training workloads.

::::::::::::::::::::::::::::::::::::: keypoints

- Vertex AI’s RAG stack = low-op, cost-predictable.
- Hugging Face = high control, high GPU cost.
- Keep data local or in GCS to manage egress and compliance.
- Always cite retrieved chunks for reproducibility and transparency.

::::::::::::::::::::::::::::::::::::::::::::::::
