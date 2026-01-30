from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs)
