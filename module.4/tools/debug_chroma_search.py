import chromadb
from src.config import SETTINGS

client = chromadb.PersistentClient(path=SETTINGS.persist_dir)

cols = client.list_collections()
print("Collections:", [c.name for c in cols])

col = client.get_collection(cols[0].name)

data = col.get(include=["documents", "metadatas"])
docs = data.get("documents", [])
metas = data.get("metadatas", [])

needle_variants = [
    "Do’s and don’ts for RAG",
    "Do's and don'ts for RAG",
    "Do’s",
    "Don'ts",
    "Combine vector + keyword search",
]

for needle in needle_variants:
    hits = []
    for i, doc in enumerate(docs):
        if doc and needle in doc:
            hits.append((i, metas[i] if i < len(metas) else {}))
    print(f"\nSearch '{needle}': {len(hits)} hits")
    for i, meta in hits[:5]:
        print(" meta:", meta)
        print(" snippet:", docs[i][:300].replace("\n", " "))
