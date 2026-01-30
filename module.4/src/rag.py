import os
import re
import math
from typing import List, Dict, Any, Tuple, Callable, Optional

from .config import SETTINGS
from .embeddings import Embedder
from .vectordb import ChromaStore
from .llm_clients import call_ollama, call_openai

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

_WORD_RE = re.compile(r"[a-z0-9]+")
_PAGE_ID_RE = re.compile(r"::PAGE_(\d+)::", re.IGNORECASE)
_CLEAN_LINE_PAGE = re.compile(r"^\[PAGE\s*\d+\]\s*$", re.IGNORECASE)
_CLEAN_LINE_NUM = re.compile(r"^\d{1,4}\s*$")
_CITE_RE = re.compile(r"\[(\d+)\]")

STOPWORDS = {
    "what", "are", "is", "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "as", "at", "by", "from", "that", "this", "it", "be", "was", "were", "you", "your", "we",
    "they", "their", "our", "why", "how", "when", "where", "which", "into", "vs", "versus"
}


class BM25Index:
    def __init__(
        self,
        ids: List[str],
        docs: List[str],
        tokenize_fn: Callable[[str], List[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.ids = ids
        self.docs = docs
        self.tokenize = tokenize_fn
        self.k1 = float(k1)
        self.b = float(b)

        self.doc_tokens: List[List[str]] = [self.tokenize(d or "") for d in docs]
        self.doc_lens: List[int] = [len(toks) for toks in self.doc_tokens]
        self.N = len(self.doc_tokens)
        self.avgdl = (sum(self.doc_lens) / max(self.N, 1)) if self.N else 1.0

        self.df: Dict[str, int] = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1

    def query(self, query: str, top_n: int = 30) -> List[Tuple[str, float]]:
        q_tokens = list(set(self.tokenize(query)))
        if not q_tokens or self.N == 0:
            return []

        idf: Dict[str, float] = {}
        for t in q_tokens:
            dft = self.df.get(t, 0)
            if dft <= 0:
                idf[t] = 0.0
            else:
                idf[t] = math.log(1 + (self.N - dft + 0.5) / (dft + 0.5))

        scored: List[Tuple[str, float]] = []
        for doc_id, toks, dl in zip(self.ids, self.doc_tokens, self.doc_lens):
            if not toks:
                continue

            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1

            s = 0.0
            for t in q_tokens:
                f = tf.get(t, 0)
                if f <= 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
                s += idf[t] * (f * (self.k1 + 1) / max(denom, 1e-9))

            if s > 0:
                scored.append((doc_id, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(int(top_n), 0)]


def _clean_context_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("’", "'")
    s = s.replace("✅", "- ").replace("🆇", "- ").replace("●", "- ").replace("✓", "- ")

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


def _normalize_for_search(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("’", "'")
    return s


def _tokenize(s: str) -> List[str]:
    s = _normalize_for_search(s).replace("'", "")
    toks = _WORD_RE.findall(s)
    return [t for t in toks if t not in STOPWORDS and len(t) >= 3]


def _build_search_text(s: str) -> str:
    return _normalize_for_search(s).replace("'", "")


def _has_query_signal(question: str, text: str) -> bool:
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return True
    st = _build_search_text(text or "")
    return any(tok in st for tok in q_tokens)


def _extract_relevant_window(question: str, text: str, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return text or ""

    q_tokens = _tokenize(question)
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


def _bm25_scores(query_tokens: List[str], doc_tokens_list: List[List[str]]) -> List[float]:
    N = len(doc_tokens_list)
    if N == 0:
        return []

    doc_lens = [len(toks) for toks in doc_tokens_list]
    avgdl = sum(doc_lens) / max(N, 1)

    df = {t: 0 for t in set(query_tokens)}
    for toks in doc_tokens_list:
        s = set(toks)
        for t in df.keys():
            if t in s:
                df[t] += 1

    idf = {}
    for t, dft in df.items():
        idf[t] = math.log(1 + (N - dft + 0.5) / (dft + 0.5)) if dft > 0 else 0.0

    scores = []
    for toks, dl in zip(doc_tokens_list, doc_lens):
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        for t in set(query_tokens):
            f = tf.get(t, 0)
            if f <= 0:
                continue
            denom = f + BM25_K1 * (1 - BM25_B + BM25_B * (dl / max(avgdl, 1e-9)))
            score += idf[t] * (f * (BM25_K1 + 1) / max(denom, 1e-9))
        scores.append(score)

    return scores


def _rerank(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    q_tokens = _tokenize(question)
    doc_tokens_list = [_tokenize(h.get("text", "") or "") for h in hits]
    bm25 = _bm25_scores(q_tokens, doc_tokens_list)

    scored: List[Tuple[float, Dict[str, Any]]] = []
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


def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    ctx_text = []
    for i, c in enumerate(contexts, start=1):
        meta = c.get("metadata", {}) or {}
        src = meta.get("source", "unknown")
        cid = meta.get("chunk_id", "unknown")

        snippet = _extract_relevant_window(question, c.get("text", "") or "", MAX_CTX_CHARS)
        snippet = _clean_context_text(snippet)

        ctx_text.append(f"[{i}] SOURCE={src} CHUNK={cid}\n{snippet}")

    ctx = "\n\n".join(ctx_text) if ctx_text else "<NO CONTEXT>"

    return f"""Answer the question using ONLY the context below.
If the answer is not in the context, say: "Not found in provided materials."

Rules:
- If the context contains relevant facts, you MUST use them to answer.
- If the question asks for a difference/contrast, write a short comparison using evidence from the context.
- Add short citations like [1], [2] based on the context item number.

Question:
{question}

Context:
{ctx}

Answer:
"""


def _sanitize_citations(answer: str, n_ctx: int) -> str:
    def repl(m: re.Match) -> str:
        k = int(m.group(1))
        return f"[{k}]" if 1 <= k <= n_ctx else ""
    return _CITE_RE.sub(repl, answer or "")


class RAGBot:
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

        self.bm25_index = BM25Index(
            ids=all_ids,
            docs=all_docs,
            tokenize_fn=_tokenize,
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

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        q_emb = self.embedder.embed([question])[0]
        vec_k = max(SETTINGS.top_k, CANDIDATE_K_VEC)

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

        primary = (kept + backfill)[: SETTINGS.top_k]

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
            print("TOP_K =", SETTINGS.top_k, "CANDIDATE_K_VEC =", vec_k, "CANDIDATE_K_LEX =", CANDIDATE_K_LEX)
            for i, h in enumerate(final, 1):
                m = h.get("metadata", {}) or {}
                print(f"  HIT{i}:", m.get("chunk_id"), m.get("source"), "dist=", h.get("distance"))

        return final

    def answer(self, question: str) -> str:
        ctxs = self.retrieve(question)
        prompt = build_prompt(question, ctxs)

        if SETTINGS.llm_provider.lower() == "openai":
            if not SETTINGS.openai_api_key:
                return "OPENAI_API_KEY is not set."
            ans = call_openai(SETTINGS.openai_model, SETTINGS.openai_api_key, prompt)
        else:
            ans = call_ollama(SETTINGS.ollama_base_url, SETTINGS.ollama_model, prompt)

        ans = (ans or "").strip()
        ans = _sanitize_citations(ans, len(ctxs))
        return ans
