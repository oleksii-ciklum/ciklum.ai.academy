Module 4 - Engineer Bonus: Multi-format RAG (PDF + audio)
=========================================================

This module includes:
1. PDF text extraction (PyMuPDF)
2. Audio extraction (ffmpeg) + transcription (faster-whisper)
3. Chunking
4. Embeddings (sentence-transformers)
5. Vector store (ChromaDB local persistent)
6. Retrieval + answer with an LLM (Ollama local)

Quick start
-----------
1. Install `ffmpeg` and ensure it's on PATH
2. Install and start Ollama: ollama serve and pull your model
3. `pip install -r requirements.txt`
4. Add files to data/raw/(see readme there for files required) then run:
- `python -m src.pipeline_ingest`
- `python -m tools.run_tests`

Outputs:
- converted pdfs and mp4s are stored in data/processed/ as txt files
- Vector DB stored in `data/processed/chroma/`
- Logs saved to `logs/run_log.txt`
