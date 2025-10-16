import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import os, json, argparse, numpy as np
from pathlib import Path
from rag import config
from rag.ingest import build_chunks
from rag.embed import Embedder
from rag.store import VectorStore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--chunk_size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--chunk_overlap", type=int, default=config.CHUNK_OVERLAP)
    args = ap.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)
    chunks = build_chunks(args.src_dir, args.chunk_size, args.chunk_overlap)

    texts = [c["text"] for c in chunks]
    meta = [{"id": c["id"], "source": c["source"], "text": c["text"]} for c in chunks]

    emb = Embedder(model_name=config.EMBED_MODEL, cache_dir=config.MODEL_CACHE_DIR)
    E = emb.encode(texts)

    # Save mirror of chunks for non-chroma backends
    mirror_chunks = Path(args.index_dir) / "fallback_chunks.jsonl"
    with open(mirror_chunks, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Build store
    chroma_dir = os.path.join(args.index_dir, "chroma")
    vs = VectorStore(persist_dir=chroma_dir, similarity=config.SIMILARITY)
    vs.add(E, meta)

    # Save index meta
    meta_out = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embed_model": config.EMBED_MODEL,
        "vector_store": config.VECTOR_STORE,
        "similarity": config.SIMILARITY,
    }
    with open(os.path.join(args.index_dir, "index_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    print(f"Indexed {len(chunks)} chunks into {args.index_dir}")

if __name__ == "__main__":
    main()
