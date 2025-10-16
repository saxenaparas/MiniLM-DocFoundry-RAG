import os, uuid
from typing import List, Dict, Any, Tuple
from .utils_io import load_corpus
from .chunk import sentence_aware_chunks

def build_chunks(src_dir: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    docs = load_corpus(src_dir)
    chunks = []
    for rel_path, text in docs:
        parts = sentence_aware_chunks(text, chunk_size, chunk_overlap)
        for i, ch in enumerate(parts):
            chunks.append({
                "id": f"{rel_path}::chunk_{i}",
                "source": rel_path,
                "text": ch
            })
    return chunks
