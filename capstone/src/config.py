import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env from capstone root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=False)


@dataclass
class Settings:
    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Embeddings
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))
    persist_dir: str = os.getenv("CHROMA_DIR", "data/processed/chroma")

    # Agent
    max_agent_iterations: int = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))
    reflection_threshold: float = float(os.getenv("REFLECTION_THRESHOLD", "3.0"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "1"))

    # Ollama generation
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
    ollama_num_thread: int = int(os.getenv("OLLAMA_NUM_THREAD", "8"))


SETTINGS = Settings()
