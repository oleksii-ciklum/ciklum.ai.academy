import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Chunk:
    id: str
    source: str
    text: str


_PAGE_RE = re.compile(r"\[PAGE\s*(\d+)\]", re.IGNORECASE)


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _split_pdf_pages(text: str) -> List[Tuple[str, str]]:
    matches = list(_PAGE_RE.finditer(text))
    if not matches:
        return [("PAGE_UNKNOWN", text)]

    pages: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        page_num = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        page_blob = text[start:end].strip()
        pages.append((f"PAGE_{page_num}", page_blob))
    return pages


def _chunk_by_paras(source: str, base_id: str, text: str, max_chars: int, overlap: int) -> List[Chunk]:
    paras = re.split(r"\n\n+", text)
    chunks: List[Chunk] = []
    buf = ""
    idx = 0

    def flush(b: str):
        nonlocal idx
        b = b.strip()
        if not b:
            return
        chunks.append(Chunk(id=f"{base_id}::{idx}", source=source, text=b))
        idx += 1

    for p in paras:
        p = p.strip()
        if not p:
            continue

        if len(buf) + len(p) + 2 <= max_chars:
            buf = buf + ("\n\n" if buf else "") + p
        else:
            flush(buf)
            tail = buf[-overlap:] if overlap and buf else ""
            buf = (tail + "\n\n" + p).strip() if tail else p

    flush(buf)
    return chunks


def chunk_text(source: str, text: str, max_chars: int = 1200, overlap: int = 200) -> List[Chunk]:
    text = clean_text(text)
    if not text:
        return []

    is_pdf = source.lower().endswith(".pdf.txt")

    chunks: List[Chunk] = []
    if is_pdf:
        pages = _split_pdf_pages(text)
        for page_label, page_text in pages:
            base_id = f"{source}::{page_label}"
            chunks.extend(_chunk_by_paras(source, base_id, page_text, max_chars=max_chars, overlap=overlap))
    else:
        base_id = f"{source}"
        chunks.extend(_chunk_by_paras(source, base_id, text, max_chars=max_chars, overlap=overlap))

    return chunks
