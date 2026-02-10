import math
import re
from typing import List, Dict, Tuple, Callable

_WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "what", "are", "is", "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "as", "at", "by", "from", "that", "this", "it", "be", "was", "were", "you", "your", "we",
    "they", "their", "our", "why", "how", "when", "where", "which", "into", "vs", "versus",
}


def tokenize(s: str) -> List[str]:
    s = (s or "").lower().replace("\u2019", "").replace("'", "")
    toks = _WORD_RE.findall(s)
    return [t for t in toks if t not in STOPWORDS and len(t) >= 3]


class BM25Index:
    def __init__(
        self,
        ids: List[str],
        docs: List[str],
        tokenize_fn: Callable[[str], List[str]] = tokenize,
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
        return scored[:max(int(top_n), 0)]


def bm25_scores_inline(query_tokens: List[str], doc_tokens_list: List[List[str]],
                        k1: float = 1.5, b: float = 0.75) -> List[float]:
    """Compute BM25 scores for a query against a list of already-tokenized docs."""
    N = len(doc_tokens_list)
    if N == 0:
        return []

    doc_lens = [len(toks) for toks in doc_tokens_list]
    avgdl = sum(doc_lens) / max(N, 1)

    df = {t: 0 for t in set(query_tokens)}
    for toks in doc_tokens_list:
        s = set(toks)
        for t in df:
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
            denom = f + k1 * (1 - b + b * (dl / max(avgdl, 1e-9)))
            score += idf[t] * (f * (k1 + 1) / max(denom, 1e-9))
        scores.append(score)

    return scores
