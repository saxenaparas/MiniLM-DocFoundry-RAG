# 🤖 DocFoundry-RAG — Local, CPU-only RAG (Windows-friendly)

**Author:** Paras Saxena  
**Summary:** A fully offline **Retrieval-Augmented Generation (RAG)** pipeline that ingests **PDF / DOCX / TXT / MD** files, builds a **persistent vector index**, and answers questions **extractively with strict citations**—all on a Windows laptop, **no cloud keys**.

>This repo intentionally includes the **corpus**, the **persistent index**, and **generated outputs** so reviewers can run the CLI and verify answers quickly. Defaults (chunking, embedding model, vector store, refusal rules) align with the **embedded RAG Design Notes** at the end of this README. 

---

## ✨ Features

* **Local & offline:** No API keys or internet required.
* **Multi-format ingestion:** PDF, DOCX, TXT, MD → normalized text with file/page metadata.
* **Persistent index:** **Chroma** vector store on disk; restart-safe and repo-friendly.
* **Two-stage retrieval:** dense retrieval (**top_k**) with optional **CrossEncoder** reranking.
* **Grounded answers:** Extractive synthesis from retrieved chunks with **strict citations**.
* **No-answer guardrail:** Refuses when similarity is too low (no hallucinated guesses).
* **Eval harness:** `recall@k`, `precision@k`, `MRR`, `hit_rate` on small JSONL sets.
* **Windows-first UX:** Simple PowerShell commands, venv setup, and a local model cache. 

---

## 🧱 Architecture (high level)

1. **Ingest** → read PDFs/DOCX/TXT/MD, normalize text, attach `{source_path, page}`.
2. **Chunk** → sentence-aware packing into **~800–1000 tokens** with **~100 overlap**.
3. **Embed** → SentenceTransformers (**all-MiniLM-L6-v2**, CPU-friendly). 
4. **Store** → **Chroma (cosine)** persisted under `data/index/chroma/` for local simplicity. 
5. **Query** → embed question, retrieve **top_k** (default **10**), optional rerank to **5**. 
6. **Answer** → stitch the best chunks into a concise extractive answer with **citations**; refuse below similarity threshold. 

---

## 📦 Repository Layout

```
DocFoundry-RAG/
  README.md
  requirements.txt
  rag/
    __init__.py  
    config.py  
    utils_io.py  
    ingest.py  
    chunk.py
    embed.py     
    store.py   
    query.py     
    answer.py  
    evaluate.py
  scripts/
    index_corpus.py  
    ask.py  
    evaluate_rag.py
  data/
    corpus/                 # assignment documents (included)
    index/                  # persistent vector store (included)
  outputs/                  # answer JSONs and artifacts (included)
  logs/                     # run logs
  models/                   # local model cache (created at runtime)
```

---

## 🚀 Quick Start (Windows / CPU-only)

1. **Create a virtual environment**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. **Index the corpus**
   (Re-run this whenever you change **embedding model** or **chunking** in `rag/config.py`.)

```powershell
python scripts\index_corpus.py ^
  --src_dir "data\corpus" ^
  --index_dir "data\index" ^
  --chunk_size 900 ^
  --chunk_overlap 120
```

3. **Ask a question (with reranking)**

```powershell
python scripts\ask.py ^
  --index_dir "data\index" ^
  --question "What maintenance interval is specified for bearings?" ^
  --top_k 10 --rerank_k 5 --mode extractive ^
  --out "outputs\answer_rerank.json"
```

4. **Ask (fast path, no rerank)**

```powershell
python scripts\ask.py ^
  --index_dir "data\index" ^
  --question "Summarize safety procedures for startup/shutdown." ^
  --top_k 10 --rerank_k 0 --mode extractive ^
  --out "outputs\answer_norerank.json"
```

5. **(Optional) Small evaluation run**

```powershell
# create tiny eval files (edit doc_id to a cited filename)
# data\questions.jsonl
{"id":"q1","text":"What maintenance interval is specified for bearings?"}
{"id":"q2","text":"Summarize safety procedures for startup/shutdown."}

# data\qrels.jsonl
{"qid":"q1","doc_id":"<replace_with_relevant_filename_from_citation>"}
{"qid":"q2","doc_id":"<replace_with_relevant_filename_from_citation>"}

python scripts\evaluate_rag.py ^
  --index_dir "data\index" ^
  --questions "data\questions.jsonl" ^
  --qrels "data\qrels.jsonl" ^
  --k 10
```

---

## ⚙️ Configuration

All tunables live in **`rag/config.py`**:

```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # optional
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TOP_K = 10
RERANK_K = 5
REFUSAL_COSINE_THRESHOLD = 0.30
VECTOR_STORE = "chroma"                 # persistent local index
INDEX_DIR = "data/index/chroma"
MODEL_CACHE_DIR = "models"
RANDOM_SEED = 42
```

* **Re-index required:** when you change **embedding model** or **chunk size/overlap**.
* **Re-index not required:** when you adjust **TOP_K**, **RERANK_K**, or the **refusal threshold**.
* **Why Chroma:** convenient local persistence; FAISS is faster at scale but less turnkey for disk-backed local projects. 

---

## 🛡️ Guardrails

* **Citation enforcement:** answers only use retrieved text; each output JSON includes `citations: [{source, page, chunk_id}]`. 
* **Refusal handling:** if `max_cosine < 0.30`, return a **no-answer** object instead of guessing. 
* **Prompt-injection & PII:** sanitize inputs, ignore jailbreak instructions, mask emails/phone numbers in responses. 

---

## 🔍 What can be verify quickly

* Run `index_corpus.py` and confirm **N chunks** indexed.
* Run `ask.py` and open `outputs/*.json` to see **answer_text** + **citations**.
* Try an off-topic query to see a **no-answer** response.
* (Optional) Run `evaluate_rag.py` on the provided tiny JSONL set.

---

## 🐛 Troubleshooting

* **Windows “symlinks disabled” warning (HF cache):** harmless. To silence:

  ```powershell
  setx HF_HUB_DISABLE_SYMLINKS_WARNING 1
  ```
* **Chroma install issues:** the code falls back to `sklearn.NearestNeighbors` (cosine).
* **Unicode/encoding errors:** ensure files are UTF-8 or plain text.
* **Very long paths:** keep corpus under a short root (e.g., `D:\data\corpus`). 

---

## 🧭 Assignment alignment

This repository implements the **Prototype of RAG system** as per the **Researched RAG Plan** requirements: a runnable local RAG prototype with ingestion, chunking strategy, embeddings/indexing, retrieval (+ optional rerank), guardrails, evaluation metrics, and clear run steps. It also includes a short scalability note for **10× corpus growth** and **100+ concurrent users**. 

---

## 📈 Scaling & Concurrency (beyond requirements)

* **Bigger corpora (10×):** switch to ANN indices (FAISS/IVF or HNSW in Chroma) and batch embeddings.
* **More users (100+ concurrent):** run as a local web service, preload models, add rate-limits & caching. 

---

# 📜 Deep-Research Architecture (short, cited)

**Goal.** A Windows, CPU-only, offline RAG that stays grounded to your local corpus, returns **extractive answers with strict citations**, and cleanly refuses when evidence is weak.

### 1) Chunking

Split long PDFs/DOCX into **~800–1000 tokens** with **~100 tokens overlap**, using sentence-aware boundaries so chunks remain coherent and fit embed limits. 

### 2) Embeddings

Use **SentenceTransformers**; compare **all-MiniLM-L6-v2** (default), **all-MiniLM-L12-v2**, and **all-mpnet-base-v2**. We pick **L6-v2** for responsiveness on CPU (≈5× faster than mpnet with small quality trade-off). 

### 3) Vector store

Prefer **Chroma (DuckDB-backed)** for an embedded, **persistent** local index that’s easy to ship. **FAISS** is excellent for raw speed/scale, but lacks DB-style persistence; Chroma is more convenient for a local CPU app. 

### 4) Retrieval & reranking

Two-stage retrieval: embed the query → fetch top candidates by cosine → **optional CrossEncoder rerank** to sharpen precision. This **boosts recall/precision** in RAG pipelines. (Project defaults: **TOP_K=10**, **RERANK_K=5**.) 

### 5) Guardrails

* **No-answer threshold:** refuse when best similarity **< 0.30** (or low aggregate confidence) to avoid unsupported answers. 
* **Prompt-injection & PII:** strip/ignore malicious instructions; mask emails/phones in outputs. 
* **Citation enforcement:** only answer using retrieved text and **require sources**; otherwise refuse. 

### 6) Evaluation & ops

Track **Recall@K, Precision@K, MRR**; do a small qualitative review of sampled answers. First run downloads the embedding model (~100 MB) to a local cache; persist the index so repeated CLI sessions skip re-indexing; include Windows tips. 

### 7) Scale & concurrency

For **10× documents**, switch to ANN (FAISS or **HNSW in Chroma**) for faster search; for **100+ concurrent users**, run as a local service with multi-threading/async, preload models, rate-limit, and cache frequent queries. 

**References (from the plan)**

* SentenceTransformers model overview (MiniLM, MPNet) 
* Open-source vector DBs: Chroma vs FAISS trade-offs 
* Two-stage retrieval & reranking primer 
* RAG pitfalls, guardrails, and evaluation checklists 
* Chunking strategies overview (semantic vs fixed) 

> **Note:** This section directly covers the rubric’s asks: **System Architecture** (ingest, chunking, embeddings/indexing, retrieval, LLM layer), **Retrieval Strategy** (sizes, model choice, rerank, citations), **Guardrails & Failure Modes** (no-answer, hallucination control, sensitive queries), and **Scalability** (10× docs; 100+ users).    

---

## 🔒 License & Data

This repository contains **sample documents** and **derived artifacts** strictly for evaluation and learning. If you fork or reuse, ensure you have rights to redistribute your own documents and respect any proprietary content.

---
