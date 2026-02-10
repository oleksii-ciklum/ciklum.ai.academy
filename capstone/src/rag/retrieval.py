import os
import re
from typing import List, Dict, Any, Optional

from ..config import SETTINGS
from .embeddings import Embedder
from .vectordb import ChromaStore
from .bm25 import BM25Index, tokenize, bm25_scores_inline

MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", "1800"))
DEBUG_RETRIEVAL = os.getenv("DEBUG_RETRIEVAL", "0") == "1"

CANDIDATE_K_VEC = int(os.getenv("CANDIDATE_K_VEC", "50"))
CANDIDATE_K_LEX = int(os.getenv("CANDIDATE_K_LEX", "30"))

VEC_WEIGHT = float(os.getenv("RERANK_VEC_WEIGHT", "0.30"))
BM25_WEIGHT = float(os.getenv("RERANK_BM25_WEIGHT", "1.20"))
PDF_BIAS = float(os.getenv("RERANK_PDF_BIAS", "0.80"))
TRANSCRIPT_PENALTY = float(os.getenv("RERANK_TRANSCRIPT_PENALTY", "0.40"))

BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

NEIGHBOR_PAGES = int(os.getenv("NEIGHBOR_PAGES", "1"))
MAX_CONTEXTS = int(os.getenv("MAX_CONTEXTS", str(SETTINGS.top_k + 6)))

_PAGE_ID_RE = re.compile(r"::PAGE_(\d+)::", re.IGNORECASE)
_CLEAN_LINE_PAGE = re.compile(r"^\[PAGE\s*\d+\]\s*$", re.IGNORECASE)
_CLEAN_LINE_NUM = re.compile(r"^\d{1,4}\s*$")
_CITE_RE = re.compile(r"\[(\d+)\]")


def _normalize_for_search(s: str) -> str:
    return (s or "").lower().replace("\u2019", "'")


def _build_search_text(s: str) -> str:
    return _normalize_for_search(s).replace("'", "")


def _has_query_signal(question: str, text: str) -> bool:
    q_tokens = set(tokenize(question))
    if not q_tokens:
        return True
    st = _build_search_text(text or "")
    return any(tok in st for tok in q_tokens)


def _extract_relevant_window(question: str, text: str, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return text or ""

    q_tokens = tokenize(question)
    if not q_tokens:
        return text[:max_chars]

    st = _build_search_text(text)
    uniq = sorted(set(q_tokens), key=lambda t: (-len(t), t))

    anchor: Optional[str] = None
    for tok in uniq:
        if st.find(tok) != -1:
            anchor = tok
            break

    if not anchor:
        return text[:max_chars]

    orig_low = _normalize_for_search(text)
    approx = orig_low.find(anchor)
    if approx == -1:
        approx = 0

    start = max(approx - max_chars // 3, 0)
    end = min(start + max_chars, len(text))
    return text[start:end]


def clean_context_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    # Replace common emoji/special chars used as bullets
    _BULLET_CHARS = "\u2705\u2713\u25cf\u2022\u25cb\u25aa\u25ba\u25b6\u2023"
    for ch in _BULLET_CHARS:
        s = s.replace(ch, "- ")
    # Remove remaining problematic Unicode (emoji, symbols above BMP) for Windows console compat
    s = re.sub(r'[\U00010000-\U0010FFFF]', '', s)

    lines = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t:
            continue
        if _CLEAN_LINE_PAGE.match(t):
            continue
        if _CLEAN_LINE_NUM.match(t):
            continue
        lines.append(t)
    return "\n".join(lines)


def _rerank(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    q_tokens = tokenize(question)
    doc_tokens_list = [tokenize(h.get("text", "") or "") for h in hits]
    bm25 = bm25_scores_inline(q_tokens, doc_tokens_list, k1=BM25_K1, b=BM25_B)

    scored = []
    for h, bm25_score in zip(hits, bm25):
        dist = float(h.get("distance", 999.0))
        vec_score = -dist

        meta = h.get("metadata", {}) or {}
        src = (meta.get("source", "") or "").lower()

        src_bias = 0.0
        if src.endswith(".pdf.txt"):
            src_bias += PDF_BIAS
        if "transcript" in src:
            src_bias -= TRANSCRIPT_PENALTY

        final = (VEC_WEIGHT * vec_score) + (BM25_WEIGHT * bm25_score) + src_bias
        scored.append((final, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored]


def sanitize_citations(answer: str, n_ctx: int) -> str:
    def repl(m: re.Match) -> str:
        k = int(m.group(1))
        return f"[{k}]" if 1 <= k <= n_ctx else ""
    return _CITE_RE.sub(repl, answer or "")


class Retriever:
    def __init__(self):
        self.embedder = Embedder(SETTINGS.embed_model)
        self.store = ChromaStore(SETTINGS.persist_dir)

        all_data = self.store.get_all()
        all_ids = all_data.get("ids", []) or []
        all_docs = all_data.get("documents", []) or []

        if all_ids and isinstance(all_ids[0], list):
            all_ids = all_ids[0]
        if all_docs and isinstance(all_docs[0], list):
            all_docs = all_docs[0]

        self.all_ids = all_ids
        self.all_docs = all_docs
        self.all_metas = all_data.get("metadatas", []) or []

        self.bm25_index = BM25Index(
            ids=all_ids,
            docs=all_docs,
            tokenize_fn=tokenize,
            k1=BM25_K1,
            b=BM25_B,
        )

    def _neighbor_page_ids(self, source: str, chunk_id: str) -> List[str]:
        m = _PAGE_ID_RE.search(chunk_id or "")
        if not m:
            return []
        p = int(m.group(1))
        out = []
        for dp in range(-NEIGHBOR_PAGES, NEIGHBOR_PAGES + 1):
            if dp == 0:
                continue
            pn = p + dp
            if pn <= 0:
                continue
            out.append(f"{source}::PAGE_{pn}::0")
        return out

    def retrieve(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = SETTINGS.top_k

        q_emb = self.embedder.embed([question])[0]
        vec_k = max(top_k, CANDIDATE_K_VEC)

        res = self.store.query(q_emb, top_k=vec_k)
        vec_docs = res.get("documents", [[]])[0]
        vec_metas = res.get("metadatas", [[]])[0]
        vec_ids = res.get("ids", [[]])[0]
        vec_dists = res.get("distances", [[]])[0]

        vec_hits: List[Dict[str, Any]] = []
        max_vec_dist = 0.0
        for doc, meta, cid, dist in zip(vec_docs, vec_metas, vec_ids, vec_dists):
            d = float(dist)
            max_vec_dist = max(max_vec_dist, d)
            vec_hits.append({"text": doc, "metadata": meta or {}, "id": cid, "distance": d})

        lex_ids_scored = self.bm25_index.query(question, top_n=CANDIDATE_K_LEX)
        lex_ids = [doc_id for doc_id, _ in lex_ids_scored]

        lex_hits: List[Dict[str, Any]] = []
        if lex_ids:
            got = self.store.get_by_ids(lex_ids)
            got_docs = got.get("documents", []) or []
            got_metas = got.get("metadatas", []) or []
            got_ids = got.get("ids", []) or []

            neutral_dist = (max_vec_dist + 0.10) if max_vec_dist > 0 else 2.0

            for doc, meta, cid in zip(got_docs, got_metas, got_ids):
                lex_hits.append({"text": doc, "metadata": meta or {}, "id": cid, "distance": neutral_dist})

        merged: Dict[str, Dict[str, Any]] = {}
        for h in vec_hits + lex_hits:
            hid = h.get("id")
            if not hid:
                continue
            if hid not in merged:
                merged[hid] = h
            else:
                if merged[hid].get("distance", 999.0) > h.get("distance", 999.0):
                    merged[hid] = h

        ranked = _rerank(question, list(merged.values()))

        kept, backfill = [], []
        for h in ranked:
            if _has_query_signal(question, h.get("text", "")):
                kept.append(h)
            else:
                backfill.append(h)

        primary = (kept + backfill)[:top_k]

        best_pdf_source = None
        for h in primary:
            meta = h.get("metadata", {}) or {}
            src = (meta.get("source", "") or "")
            if src.lower().endswith(".pdf.txt"):
                best_pdf_source = src
                break

        neighbor_ids: List[str] = []
        if best_pdf_source:
            for h in primary:
                meta = h.get("metadata", {}) or {}
                src = (meta.get("source", "") or "")
                if src != best_pdf_source:
                    continue
                cid = (meta.get("chunk_id", "") or "")
                neighbor_ids.extend(self._neighbor_page_ids(src, cid))

        extras: List[Dict[str, Any]] = []
        if neighbor_ids:
            got = self.store.get_by_ids(neighbor_ids)
            extra_docs = got.get("documents", []) or []
            extra_metas = got.get("metadatas", []) or []
            extra_ids = got.get("ids", []) or []
            for doc, meta, cid in zip(extra_docs, extra_metas, extra_ids):
                extras.append({"text": doc, "metadata": meta or {}, "id": cid, "distance": 999.0})

        seen = set()
        final: List[Dict[str, Any]] = []
        for h in primary + extras:
            hid = h.get("id")
            if not hid or hid in seen:
                continue
            seen.add(hid)
            final.append(h)
            if len(final) >= MAX_CONTEXTS:
                break

        if DEBUG_RETRIEVAL:
            print(f"[retrieval] TOP_K={top_k} VEC_K={vec_k} LEX_K={CANDIDATE_K_LEX}")
            for i, h in enumerate(final, 1):
                m = h.get("metadata", {}) or {}
                print(f"  HIT{i}: {m.get('chunk_id')} src={m.get('source')} dist={h.get('distance')}")

        return final

    def get_sources(self) -> List[str]:
        """Return list of unique source document names."""
        sources = set()
        for meta in self.all_metas:
            src = (meta or {}).get("source", "")
            if src:
                sources.add(src)
        return sorted(sources)

    def get_page_text(self, source: str, page: int) -> str:
        """Return text of a specific page from a source document."""
        target_id_prefix = f"{source}::PAGE_{page}::"
        texts = []
        for cid, doc in zip(self.all_ids, self.all_docs):
            if cid.startswith(target_id_prefix):
                texts.append(doc or "")
        if not texts:
            return f"No content found for source='{source}', page={page}."
        return "\n\n".join(texts)

    def format_contexts(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """Format retrieved contexts into a numbered string for LLM prompts."""
        parts = []
        for i, c in enumerate(contexts, start=1):
            meta = c.get("metadata", {}) or {}
            src = meta.get("source", "unknown")
            cid = meta.get("chunk_id", "unknown")

            snippet = _extract_relevant_window(question, c.get("text", "") or "", MAX_CTX_CHARS)
            snippet = clean_context_text(snippet)

            parts.append(f"[{i}] SOURCE={src} CHUNK={cid}\n{snippet}")

        return "\n\n".join(parts) if parts else "<NO CONTEXT>"
