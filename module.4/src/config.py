import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=False)

@dataclass
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = int(os.getenv("TOP_K", "5"))
    persist_dir: str = os.getenv("CHROMA_DIR", "data/processed/chroma")

SETTINGS = Settings()
