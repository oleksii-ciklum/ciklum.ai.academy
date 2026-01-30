from pathlib import Path
from tqdm import tqdm
from .config import SETTINGS
from .pdf_extract import save_pdf_text
from .audio_transcribe import mp4_to_wav, transcribe_wav
from .chunking import chunk_text
from .embeddings import Embedder
from .vectordb import ChromaStore

RAW = Path("data/raw")
DOCS = Path("data/processed/docs")

def ensure_inputs():
    needed = ["rag_intro.pdf", "databases_for_genai.pdf", "rag_intro.mp4", "databases_for_genai.mp4"]
    missing = [n for n in needed if not (RAW/n).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in data/raw: {missing}")

def main():
    ensure_inputs()
    DOCS.mkdir(parents=True, exist_ok=True)

    pdfs = [
        (RAW/"rag_intro.pdf", DOCS/"rag_intro.pdf.txt"),
        (RAW/"databases_for_genai.pdf", DOCS/"databases_for_genai.pdf.txt"),
    ]
    for pdf, out in pdfs:
        print(f"Extracting PDF: {pdf.name}")
        save_pdf_text(pdf, out)

    audios = [
        (RAW/"rag_intro.mp4", DOCS/"rag_intro.wav", DOCS/"rag_intro.transcript.txt"),
        (RAW/"databases_for_genai.mp4", DOCS/"databases_for_genai.wav", DOCS/"databases_for_genai.transcript.txt"),
    ]
    for mp4, wav, out in audios:
        if not wav.exists():
            print(f"Extracting audio: {mp4.name}")
            mp4_to_wav(mp4, wav)
        print(f"Transcribing: {wav.name}")
        transcribe_wav(wav, out, model_name="small")

    sources = list(DOCS.glob("*.txt"))
    chunks = []
    for p in sources:
        chunks.extend(chunk_text(p.name, p.read_text(encoding="utf-8", errors="ignore")))
    print(f"Total chunks: {len(chunks)}")

    embedder = Embedder(SETTINGS.embed_model)
    store = ChromaStore(SETTINGS.persist_dir)

    batch = 64
    for i in tqdm(range(0, len(chunks), batch), desc="Embedding"):
        b = chunks[i:i+batch]
        ids = [c.id for c in b]
        docs = [c.text for c in b]
        metas = [{"source": c.source, "chunk_id": c.id} for c in b]
        embs = embedder.embed(docs)
        store.upsert(ids, embs, docs, metas)

    print("Done. Chroma DB ready.")

if __name__ == "__main__":
    main()
