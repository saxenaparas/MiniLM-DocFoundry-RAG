import os, math, hashlib, numpy as np
from typing import List, Optional

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.backend = None
        self.dim = 384  # default MiniLM dim; used by hashing fallback
        self._init_backend()

    def _init_backend(self):
        # Try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.backend = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            # Update dim from model config if present
            try:
                self.dim = self.backend.get_sentence_embedding_dimension()
            except Exception:
                pass
            return
        except Exception:
            self.backend = None

        # Try sklearn HashingVectorizer as middle fallback
        try:
            from sklearn.feature_extraction.text import HashingVectorizer
            self.backend = ("sk_hash", HashingVectorizer(n_features=self.dim, alternate_sign=False, norm="l2"))
            return
        except Exception:
            self.backend = None

        # Last resort: pure-NumPy hashing to fixed dim
        self.backend = ("np_hash", None)

    def encode(self, texts: List[str]) -> np.ndarray:
        if hasattr(self.backend, "encode"):
            return np.asarray(self.backend.encode(texts, convert_to_numpy=True))
        kind = self.backend[0] if isinstance(self.backend, tuple) else None
        if kind == "sk_hash":
            hv = self.backend[1]
            X = hv.transform(texts)  # sparse
            return X.toarray().astype("float32")
        elif kind == "np_hash":
            # Very simple token hashing bag-of-words
            mat = np.zeros((len(texts), self.dim), dtype="float32")
            for i, t in enumerate(texts):
                for tok in t.split():
                    h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                    idx = h % self.dim
                    mat[i, idx] += 1.0
                # L2 normalize
                norm = np.linalg.norm(mat[i])
                if norm > 0:
                    mat[i] /= norm
            return mat
        else:
            # Should not happen, but safeguard
            return np.zeros((len(texts), self.dim), dtype="float32")
