"""Data ingestion pipeline: extract PDFs + audio, chunk, embed, store in ChromaDB."""

import sys
from pathlib import Path

# Ensure capstone root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm
from src.config import SETTINGS
from src.rag.chunking import chunk_text
from src.rag.embeddings import Embedder
from src.rag.vectordb import ChromaStore

RAW = Path("data/raw")
DOCS = Path("data/processed/docs")

EXPECTED_PDFS = ["rag_intro.pdf", "databases_for_genai.pdf"]
EXPECTED_VIDEOS = ["rag_intro.mp4", "databases_for_genai.mp4"]


def ensure_inputs():
    needed = EXPECTED_PDFS + EXPECTED_VIDEOS
    missing = [n for n in needed if not (RAW / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {RAW}: {missing}\n"
            "Place the required PDFs and MP4s in data/raw/ before running ingestion."
        )


def extract_pdfs():
    from src.rag.pdf_extract import save_pdf_text

    results = []
    for name in EXPECTED_PDFS:
        pdf_path = RAW / name
        out_path = DOCS / f"{name}.txt"
        print(f"Extracting PDF: {name}")
        save_pdf_text(pdf_path, out_path)
        results.append(out_path)
    return results


def extract_and_transcribe_audio():
    from src.rag.audio_transcribe import mp4_to_wav, transcribe_wav

    results = []
    for name in EXPECTED_VIDEOS:
        stem = Path(name).stem
        mp4_path = RAW / name
        wav_path = DOCS / f"{stem}.wav"
        out_path = DOCS / f"{stem}.transcript.txt"

        if not wav_path.exists():
            print(f"Extracting audio: {name}")
            mp4_to_wav(mp4_path, wav_path)

        print(f"Transcribing: {wav_path.name}")
        transcribe_wav(wav_path, out_path, model_name="small")
        results.append(out_path)
    return results


def main():
    ensure_inputs()
    DOCS.mkdir(parents=True, exist_ok=True)

    extract_pdfs()
    extract_and_transcribe_audio()

    # Chunk all text files
    sources = list(DOCS.glob("*.txt"))
    chunks = []
    for p in sources:
        chunks.extend(chunk_text(p.name, p.read_text(encoding="utf-8", errors="ignore")))
    print(f"Total chunks: {len(chunks)}")

    # Embed and store
    embedder = Embedder(SETTINGS.embed_model)
    store = ChromaStore(SETTINGS.persist_dir)

    batch = 64
    for i in tqdm(range(0, len(chunks), batch), desc="Embedding"):
        b = chunks[i:i + batch]
        ids = [c.id for c in b]
        docs = [c.text for c in b]
        metas = [{"source": c.source, "chunk_id": c.id} for c in b]
        embs = embedder.embed(docs)
        store.upsert(ids, embs, docs, metas)

    print("Done. ChromaDB ready at:", SETTINGS.persist_dir)


if __name__ == "__main__":
    main()
