import re
from typing import List

def split_sentences(text: str) -> List[str]:
    # Minimal sentence split (no NLTK): split on ., !, ?, or newlines while preserving simple structure
    # Avoid empty fragments
    parts = re.split(r'(?<=[\.\!\?])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]

def sentence_aware_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    sents = split_sentences(text)
    chunks = []
    buffer = ""
    for s in sents:
        # Token proxy: use whitespace words as token-proxy
        if not buffer:
            buffer = s
        else:
            candidate = buffer + " " + s
            if len(candidate.split()) <= chunk_size:
                buffer = candidate
            else:
                chunks.append(buffer)
                # overlap: take last 'chunk_overlap' words as seed
                if chunk_overlap > 0:
                    words = buffer.split()
                    seed = " ".join(words[-chunk_overlap:]) if len(words) > chunk_overlap else buffer
                    buffer = (seed + " " + s).strip()
                else:
                    buffer = s
    if buffer:
        chunks.append(buffer)
    return chunks
