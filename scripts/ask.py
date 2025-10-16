import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import os, json, argparse, numpy as np
from rag import config
from rag.embed import Embedder
from rag.store import VectorStore
from rag.query import topk_from_store, rerank_if_available
from rag.answer import sanitize_question, build_extractive_answer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--top_k", type=int, default=config.TOP_K)
    ap.add_argument("--rerank_k", type=int, default=config.RERANK_K)
    ap.add_argument("--mode", choices=["extractive"], default="extractive")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    q = sanitize_question(args.question)

    # Load mirrors
    # We still create a store object to allow chroma/sklearn/numpy querying
    chroma_dir = os.path.join(args.index_dir, "chroma")
    store = VectorStore(persist_dir=chroma_dir, similarity=config.SIMILARITY)

    # Embed query
    emb = Embedder(model_name=config.EMBED_MODEL, cache_dir=config.MODEL_CACHE_DIR)
    qE = emb.encode([q])

    # Retrieve
    hits = topk_from_store(store, qE, top_k=args.top_k, metas=[])

    # Rerank if available/asked
    hits = rerank_if_available(q, hits, args.rerank_k)

    # Build answer (extractive)
    out = {
        "question": args.question,
        "defaults": {
            "CHUNK_SIZE": config.CHUNK_SIZE,
            "CHUNK_OVERLAP": config.CHUNK_OVERLAP,
            "EMBED_MODEL": config.EMBED_MODEL,
            "TOP_K": args.top_k,
            "RERANK_K": args.rerank_k,
            "REFUSAL_COSINE_THRESHOLD": config.REFUSAL_COSINE_THRESHOLD,
            "VECTOR_STORE": config.VECTOR_STORE,
            "SIMILARITY": config.SIMILARITY,
        },
    }
    ans = build_extractive_answer(args.question, hits, config.REFUSAL_COSINE_THRESHOLD)
    out.update(ans)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
