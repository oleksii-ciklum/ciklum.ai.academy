from typing import List, Dict, Any
from pathlib import Path
import chromadb


class ChromaStore:
    def __init__(self, persist_dir: str, collection_name: str = "academy_docs"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.col = self.client.get_or_create_collection(name=collection_name)

    def upsert(self, ids: List[str], embeddings, documents: List[str], metadatas: List[Dict[str, Any]]):
        self.col.upsert(ids=ids, embeddings=embeddings.tolist(), documents=documents, metadatas=metadatas)

    def query(self, query_embedding, top_k: int = 5):
        return self.col.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        ids = list(dict.fromkeys(ids))
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}
        return self.col.get(ids=ids, include=["documents", "metadatas"])

    def get_all(self, batch_size: int = 1000) -> Dict[str, Any]:
        all_ids: List[str] = []
        all_docs: List[str] = []
        all_metas: List[Dict[str, Any]] = []

        offset = 0
        while True:
            res = self.col.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
            ids = res.get("ids", []) or []
            if not ids:
                break

            docs = res.get("documents", []) or []
            metas = res.get("metadatas", []) or []

            all_ids.extend(ids)
            all_docs.extend(docs)
            all_metas.extend(metas)

            offset += len(ids)
            if len(ids) < batch_size:
                break

        return {"ids": all_ids, "documents": all_docs, "metadatas": all_metas}
