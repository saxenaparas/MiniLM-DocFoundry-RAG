import os, re, json, pathlib
from typing import List, Tuple

def read_text_file(path: pathlib.Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""

def read_pdf(path: pathlib.Path) -> str:
    try:
        import PyPDF2
        txt = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                txt += page.extract_text() or ""
        return txt
    except Exception:
        return ""

def read_docx(path: pathlib.Path) -> str:
    try:
        import docx
        doc = docx.Document(str(path))
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def load_corpus(src_dir: str) -> List[Tuple[str, str]]:
    """Return list of (relative_path, text) for supported files."""
    base = pathlib.Path(src_dir)
    out = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        text = ""
        if ext in [".txt", ".md"]:
            text = read_text_file(p)
        elif ext == ".pdf":
            text = read_pdf(p)
        elif ext in [".docx"]:
            text = read_docx(p)
        else:
            # unsupported -> skip (stay purely local)
            text = ""
        if text.strip():
            out.append((str(p.relative_to(base)), text))
    return out
