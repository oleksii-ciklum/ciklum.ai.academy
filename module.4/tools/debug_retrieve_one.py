import os
import sys
from src.rag import RAGBot
from src.config import SETTINGS

def main():
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        print("Usage: python debug_retrieve_one.py \"your question\"")
        return

    bot = RAGBot()
    ctxs = bot.retrieve(q)

    print("TOP_K =", SETTINGS.top_k, "CANDIDATE_K_VEC =", os.getenv("CANDIDATE_K_VEC"), "CANDIDATE_K_LEX =", os.getenv("CANDIDATE_K_LEX"))
    for i, c in enumerate(ctxs, 1):
        meta = c.get("metadata", {}) or {}
        print(f"  HIT{i}:", meta.get("chunk_id"), meta.get("source"), "dist=", c.get("distance"))

    print("QUESTION:", q)
    print("HITS:", len(ctxs))
    for i, c in enumerate(ctxs, 1):
        meta = c.get("metadata", {}) or {}
        print("\n--- HIT", i, "---")
        print("source:", meta.get("source"))
        print("chunk_id:", meta.get("chunk_id"))
        print("distance:", c.get("distance"))
        text = (c.get("text") or "").replace("\n", " ")
        print("text_snippet:", text[:500])

if __name__ == "__main__":
    main()
