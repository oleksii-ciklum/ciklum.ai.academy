import os
import requests

DEFAULT_SYSTEM = (
    "You are a careful assistant. Use ONLY the provided context. "
    "If the context contains relevant information that answers the question, you MUST answer using it. "
    "Only reply exactly: \"Not found in provided materials.\" if the context contains no relevant information."
)

def call_ollama(base_url: str, model: str, prompt: str) -> str:
    url = base_url.rstrip("/") + "/api/chat"

    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    system = os.getenv("OLLAMA_SYSTEM", DEFAULT_SYSTEM)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }

    r = requests.post(url, json=payload, timeout=(10, 600))
    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content", "")

def call_openai(model: str, api_key: str, prompt: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer using only the provided context."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
