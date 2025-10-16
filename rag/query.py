import numpy as np
from typing import List, Dict, Any, Tuple

def rerank_if_available(query: str, candidates: List[Dict[str,Any]], rerank_k: int) -> List[Dict[str,Any]]:
    if rerank_k <= 0 or len(candidates) == 0:
        return candidates
    rerank_k = min(rerank_k, len(candidates))
    try:
        from sentence_transformers import CrossEncoder
        # use a small cross-encoder if available
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, c["text"]) for c in candidates[:rerank_k]]
        scores = model.predict(pairs).tolist()
        rescored = [{"score": float(s), **candidates[i]} for i, s in enumerate(scores)]
        rescored.sort(key=lambda x: x["score"], reverse=True)
        # keep rescored top region, then append the remainder (unreranked)
        rest = candidates[rerank_k:]
        return rescored + rest
    except Exception:
        return candidates

def topk_from_store(store, q_embed, top_k: int, metas: List[Dict[str,Any]]):
    idxs, sims, metas_out = store.query(q_embed, top_k=top_k)
    # The store already returns metadatas, but keep signature consistent
    out = []
    if metas_out and len(metas_out[0])>0:
        # using store-provided metas
        for j, sim in zip(metas_out[0], sims[0]):
            c = dict(j)
            c["similarity"] = float(sim)
            out.append(c)
    return out
