import os, json, numpy as np
from typing import List, Dict, Any, Optional, Tuple

class VectorStore:
    """Chroma (persistent) preferred, with fallbacks: sklearn NearestNeighbors, then NumPy brute-force."""
    def __init__(self, persist_dir: str, similarity: str = "cosine"):
        self.persist_dir = persist_dir
        self.similarity = similarity
        self.backend = None
        self.collection = None
        self.meta_path = os.path.join(os.path.dirname(persist_dir), "fallback_index_meta.json")
        self.emb_path = os.path.join(os.path.dirname(persist_dir), "fallback_embeddings.npy")
        self.chunks_path = os.path.join(os.path.dirname(persist_dir), "fallback_chunks.jsonl")
        self._init_backend()

    def _init_backend(self):
        # Try Chroma
        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(anonymized_telemetry=False))
            self.collection = client.get_or_create_collection(name="paras_rag", metadata={"hnsw:space": "cosine"})
            self.backend = "chroma"
            return
        except Exception:
            self.backend = None
            self.collection = None

        # Try sklearn
        try:
            from sklearn.neighbors import NearestNeighbors
            self.nn = NearestNeighbors(metric="cosine")
            self.backend = "sklearn"
            return
        except Exception:
            self.backend = None

        # Last resort: numpy brute-force
        self.backend = "numpy"

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        if ids is None:
            ids = [f"id_{i}" for i in range(len(metadatas))]
        if self.backend == "chroma":
            self.collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=[m.get("text","") for m in metadatas])
        else:
            # Persist to fallback files
            os.makedirs(os.path.dirname(self.emb_path), exist_ok=True)
            # Append mode handling
            prev_E = None
            prev_meta = []
            prev_ids = []
            if os.path.exists(self.emb_path):
                prev_E = np.load(self.emb_path)
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    prev_meta = [json.loads(line) for line in f]
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    prev_ids = meta.get("ids", [])
            E = embeddings if prev_E is None else np.vstack([prev_E, embeddings])
            np.save(self.emb_path, E)
            with open(self.chunks_path, "a", encoding="utf-8") as f:
                for m in metadatas:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            ids_all = prev_ids + ids
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump({"ids": ids_all, "backend": self.backend, "similarity": self.similarity}, f)

            # Fit sklearn if available
            if self.backend == "sklearn":
                from sklearn.neighbors import NearestNeighbors
                self.nn.fit(E)

    def _cosine_sim(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # Normalize rows
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return A_norm @ B_norm.T

    def query(self, query_embeddings: np.ndarray, top_k: int = 5) -> Tuple[List[List[int]], List[List[float]], List[List[Dict[str,Any]]]]:
        if self.backend == "chroma":
            res = self.collection.query(query_embeddings=query_embeddings.tolist(), n_results=top_k, include=["metadatas", "distances"])
            # Chroma distances for "cosine" are actually L2/cosine distances; convert to similarity (approx. 1 - distance)
            all_idxs = []
            all_sims = []
            all_metas = []
            for i in range(len(query_embeddings)):
                metas = res.get("metadatas", [[]])[i]
                dists = res.get("distances", [[]])[i]
                sims = [1.0 - float(d) for d in dists]
                idxs = list(range(len(metas)))
                all_idxs.append(idxs)
                all_sims.append(sims)
                all_metas.append(metas)
            return all_idxs, all_sims, all_metas

        # Fallback: load persisted arrays & metadata
        if not os.path.exists(self.emb_path):
            return [[]], [[]], [[]]
        E = np.load(self.emb_path)
        metas = []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            metas = [json.loads(line) for line in f]

        if self.backend == "sklearn":
            # Ensure NN is fitted on loaded embeddings
            from sklearn.neighbors import NearestNeighbors
            self.nn.fit(E)
            # sklearn returns distances; convert to similarity ~ 1 - dist
            dists, idxs = self.nn.kneighbors(query_embeddings, n_neighbors=min(top_k, E.shape[0]))
            sims = (1.0 - dists).tolist()
            out_metas = [[metas[j] for j in row] for row in idxs]
            return idxs.tolist(), sims, out_metas

        # Numpy brute force
        sims_full = self._cosine_sim(query_embeddings, E)  # (Q, N)
        all_idxs = []
        all_sims = []
        all_metas = []
        for i in range(sims_full.shape[0]):
            row = sims_full[i]
            order = np.argsort(-row)[:top_k]
            all_idxs.append(order.tolist())
            all_sims.append(row[order].tolist())
            all_metas.append([metas[j] for j in order])
        return all_idxs, all_sims, all_metas
