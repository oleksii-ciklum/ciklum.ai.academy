import os
import json
import re
import requests
from typing import List, Dict, Optional

from ..config import SETTINGS


def call_ollama(prompt: str, system: str = "", messages: Optional[List[Dict[str, str]]] = None) -> str:
    """Call Ollama API. If messages is provided, use them directly; otherwise build from prompt."""
    url = SETTINGS.ollama_base_url.rstrip("/") + "/api/chat"

    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": SETTINGS.ollama_model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": SETTINGS.ollama_num_ctx,
            "num_predict": SETTINGS.ollama_num_predict,
            "temperature": SETTINGS.ollama_temperature,
            "num_thread": SETTINGS.ollama_num_thread,
        },
    }

    r = requests.post(url, json=payload, timeout=(10, 600))
    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content", "")


def call_openai(prompt: str, system: str = "", messages: Optional[List[Dict[str, str]]] = None) -> str:
    """Call OpenAI API."""
    if not SETTINGS.openai_api_key:
        return "ERROR: OPENAI_API_KEY is not set."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {SETTINGS.openai_api_key}", "Content-Type": "application/json"}

    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": SETTINGS.openai_model,
        "messages": messages,
        "temperature": SETTINGS.ollama_temperature,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_llm(prompt: str, system: str = "", messages: Optional[List[Dict[str, str]]] = None) -> str:
    """Dispatch to configured LLM provider."""
    if SETTINGS.llm_provider.lower() == "openai":
        return call_openai(prompt, system, messages)
    return call_ollama(prompt, system, messages)


def call_llm_json(prompt: str, system: str = "", messages: Optional[List[Dict[str, str]]] = None) -> dict:
    """Call LLM and attempt to parse JSON from the response.
    Falls back to extracting JSON from markdown code blocks if needed.
    """
    raw = call_llm(prompt, system, messages)

    # Try direct parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from ```json ... ``` blocks
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # Try finding first { ... } or [ ... ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = raw.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i + 1])
                    except (json.JSONDecodeError, TypeError):
                        break

    # Last resort: return raw text wrapped in a dict
    return {"_raw": raw, "_parse_error": True}
