import re, json
from typing import List, Dict, Any

def sanitize_question(q: str) -> str:
    # Very basic PI/injection mitigation: remove directives like "ignore previous…" etc.
    q = re.sub(r'(?i)ignore.*previous.*instructions', '', q)
    q = re.sub(r'(?i)system:.*', '', q)
    return q.strip()

def mask_pii(text: str) -> str:
    # mask simple emails and phone numbers
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    return text

def build_extractive_answer(question: str, hits: List[Dict[str,Any]], refusal_threshold: float) -> Dict[str,Any]:
    if not hits:
        return {"answer": None, "refusal": True, "reason": "no_hits", "sources": []}

    best = max(h["similarity"] for h in hits if "similarity" in h) if hits else 0.0
    if best < refusal_threshold:
        return {"answer": None, "refusal": True, "reason": f"similarity<{refusal_threshold}", "sources": hits[:3]}

    # Extractive heuristic: concatenate top 2-3 chunks as evidence
    extracted = " ".join([h.get("text","") for h in hits[:3]])
    extracted = mask_pii(extracted)
    return {"answer": extracted, "refusal": False, "sources": hits[:5]}
