from src.config import SETTINGS
import chromadb

COLLECTION = "academy_docs"
CHUNK_ID = "databases_for_genai.pdf.txt::5"

client = chromadb.PersistentClient(path=SETTINGS.persist_dir)
col = client.get_collection(COLLECTION)

res = col.get(ids=[CHUNK_ID], include=["documents", "metadatas"])
doc = res["documents"][0]
meta = res["metadatas"][0]

print("META:", meta)
print("CHARS:", len(doc))
print("\n--- HEAD (first 1200 chars) ---\n")
print(doc[:1200])
print("\n--- TAIL (last 1200 chars) ---\n")
print(doc[-1200:])
print("\nPAGE_MARKERS_FOUND:", doc.count("[PAGE"))
