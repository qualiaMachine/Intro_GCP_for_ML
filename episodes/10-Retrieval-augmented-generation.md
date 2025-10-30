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
However, this requires a GPU or large CPU VM (e.g., `n1-standard-8` + `T4`) and manual model management.  
Vertex AI’s managed models (`text-embedding-004`, `gemini-2.5-flash-001`) are cost-optimized and scalable — better for workshops or low-ops setups.



## Step 1: Setup environment

```python
!pip install --quiet --upgrade google-cloud-aiplatform google-cloud-storage vertexai pypdf scikit-learn pandas
```

**Cost note:** Installing packages is free; you're only billed for VM runtime.

### Initialize project

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

```python
import zipfile, pathlib, re, pandas as pd
from pypdf import PdfReader

ZIP_PATH = pathlib.Path("Intro_GCP_VertexAI/data/pdfs_bundle.zip")
DOC_DIR = pathlib.Path("./docs")
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

import pandas as pd
corpus_df = pd.DataFrame(rows)
print(len(corpus_df), "chunks created")
```

**Cost note:** Only VM runtime applies. Chunk size affects future embedding cost.



## Step 3: Embed text using Vertex AI

```python
from vertexai.language_models import TextEmbeddingModel
import numpy as np

model = TextEmbeddingModel.from_pretrained("text-embedding-004")

def get_embeddings(texts):
    vecs = []
    for batch in [texts[i:i+32] for i in range(0, len(texts), 32)]:
        resp = model.get_embeddings(batch)
        for r in resp: vecs.append(r.values)
    return np.array(vecs, dtype="float32")

emb_matrix = get_embeddings(corpus_df.text.tolist())
print("Embeddings shape:", emb_matrix.shape)
```

**Cost note:** This is the first paid step — roughly $0.0001–0.0002 per 1k tokens.  
Cache embeddings locally or in GCS to avoid recharging later.



## Step 4: Retrieve relevant chunks

```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(metric="cosine", n_neighbors=5)
nn.fit(emb_matrix)

def retrieve(query):
    q_vec = model.get_embeddings([query])[0].values
    dist, idx = nn.kneighbors([q_vec], return_distance=True)
    df = corpus_df.iloc[idx[0]].copy()
    df["similarity"] = 1 - dist[0]
    return df.sort_values("similarity", ascending=False)
```

**Cost note:** Retrieval is free (runs locally). Each new query triggers one embedding call.



## Step 5: Generate answers using Gemini

```python
from vertexai.generative_models import GenerativeModel

gmodel = GenerativeModel("gemini-2.5-flash-001")

def ask(query, top_k=5):
    hits = retrieve(query).head(top_k)
    context = "\n\n".join([f"[{r.doc}#chunk-{r.chunk_id}] {r.text}" for _, r in hits.iterrows()])
    prompt = f"You are a sustainability analyst. Use only the following context to answer the question.\n\n{context}\n\nQ: {query}\nA:"
    ans = gmodel.generate_content(prompt)
    return ans.text

print(ask("What water usage (WUE) is reported for model training?"))
```

**Cost note:** Gemini Flash costs ~$0.00005 per 1k input tokens and ~$0.0002 per 1k output tokens.  
To stay under $1 for the workshop, limit to short prompts and <100 queries.



## Step 6: (Optional) Hugging Face local substitution

To avoid managed API costs, you can install and run local models:

```python
!pip install --quiet sentence-transformers transformers accelerate bitsandbytes
```

```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

embedder = SentenceTransformer("all-MiniLM-L6-v2")
local_embs = embedder.encode(corpus_df.text.tolist())

tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct")
mdl = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct", device_map="auto", load_in_4bit=True)

pipe = pipeline("text-generation", model=mdl, tokenizer=tok)

def local_answer(query):
    hits = retrieve(query).head(3)
    context = "\n\n".join([f"[{r.doc}] {r.text}" for _, r in hits.iterrows()])
    prompt = f"Answer based only on this evidence:\n{context}\nQ: {query}\nA:"
    return pipe(prompt, max_new_tokens=150)[0]["generated_text"]
```

**Trade-offs:**  
- Hugging Face embeddings and generation are free but require powerful hardware.  
- Vertex AI managed models cost less for small workloads and scale to large datasets automatically.  
- No prebuilt “Hugging Face RAG” image exists on GCP; use `PyTorch` Workbench image and `pip install` as above.



## Step 7: Cost summary

| Step | Resource | Example Component | Cost Driver | Typical Range |
|------|-----------|-------------------|--------------|----------------|
| VM runtime | Vertex AI Workbench | `n1-standard-4` | Uptime (hourly) | ~$0.20/hr |
| Embeddings | text-embedding-004 | Managed API | Tokens embedded | ~$0.10 / 1M tokens |
| Retrieval | Local NN | CPU only | None | Free |
| Generation | gemini-2.5-flash-001 | Managed API | Input/output tokens | ~$0.25 / 1M tokens |
| Hugging Face alt | T4 VM | Local model inference | GPU uptime | ~$0.35/hr + egress |



## Key takeaways

- Use **Vertex AI managed embeddings** and **Gemini Flash** for lightweight, cost-controlled RAG.
- Cache embeddings; reusing them saves most cost.
- For open alternatives, use Hugging Face models on GPU VMs (higher cost, more control).
- This workflow generalizes to any retrieval task — not just sustainability papers.
- GCP’s managed tools lower barrier for experimentation while keeping enterprise security and IAM intact.

::::::::::::::::::::::::::::::::::::: keypoints

- Vertex AI’s RAG stack = low-op, cost-predictable.  
- Hugging Face = high control, high GPU cost.  
- Keep data local or in GCS to manage egress and compliance.  
- Always cite retrieved chunks for reproducibility and transparency.

::::::::::::::::::::::::::::::::::::::::::::::::
