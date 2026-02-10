# Module 4: Multi-Format RAG Chatbot (PDF + Audio)

A functional RAG chatbot built from scratch that ingests a knowledge base from multiple file formats (PDFs and audio lecture recordings), stores it in a vector database, and answers questions using a local LLM with cited sources.

## Assignment Requirements Coverage

| Requirement | Status | Implementation |
|---|---|---|
| **Load and process PDF data** | Done | `src/pdf_extract.py` - PyMuPDF extracts text page-by-page with `[PAGE N]` markers |
| **Speech-to-text transcription** | Done | `src/audio_transcribe.py` - faster-whisper (small model) with ffmpeg for audio extraction |
| **Chunk the text** | Done | `src/chunking.py` - paragraph-based splitting with configurable max size and overlap |
| **Embed and store in vector DB** | Done | `src/embeddings.py` + `src/vectordb.py` - sentence-transformers → ChromaDB (persistent) |
| **Retrieve and generate** | Done | `src/rag.py` - hybrid retrieval (vector + BM25) with reranking, then LLM answer generation |
| **Test with 3+ questions** | Done | `tools/run_tests.py` - runs 3 required test questions, logs answers to file |
| **requirements.txt** | Done | All dependencies listed |
| **Answer log** | Done | Saved to `logs/run_log.txt` |

## Architecture

```
Source Files (PDF + MP4)
        |
        v
[PDF Extract] ──> [Text Files] ──> [Chunking] ──> [Embeddings] ──> [ChromaDB]
[Audio Extract + Transcribe] ──────────┘                               |
                                                                       v
User Question ──> [Embed Query] ──> [Vector Search] ──> [Rerank] ──> [Build Prompt] ──> [LLM] ──> Answer
                                    [BM25 Search] ──────────┘
```

### Key Design Decisions

- **Hybrid retrieval**: Vector cosine similarity + custom BM25 lexical search, merged and reranked
- **Source-type bias**: PDF chunks are boosted over transcripts in reranking (transcripts are noisier)
- **Neighbor page expansion**: Adjacent PDF pages are pulled in for context continuity
- **Local-first**: Ollama (local LLM) as default - no API keys needed; OpenAI as optional fallback
- **Paragraph-aware chunking**: Splits on paragraph boundaries rather than fixed character counts

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally
- ffmpeg installed and on PATH (for audio extraction from MP4)

### Installation

```bash
cd module.4
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env if needed (defaults work with local Ollama)
```

Pull the Ollama model:
```bash
ollama pull llama3.1:latest
```

### Data Preparation

Place these files in `data/raw/`:
- `rag_intro.pdf`
- `databases_for_genai.pdf`
- `rag_intro.mp4`
- `databases_for_genai.mp4`

### Ingestion

```bash
python -m src.pipeline_ingest
```

This runs the full pipeline:
1. Extracts text from PDFs (page by page with markers)
2. Extracts audio from MP4 files via ffmpeg
3. Transcribes audio using faster-whisper (small model)
4. Chunks all text with paragraph-aware splitting (1200 chars, 200 overlap)
5. Generates embeddings (sentence-transformers/all-MiniLM-L6-v2)
6. Stores everything in ChromaDB (persistent local storage)

## Usage

### Run Test Questions

```bash
python -m tools.run_tests
```

Runs the 3 required test questions and saves answers to `logs/run_log.txt`:
1. "What are the production 'Do's' for RAG?"
2. "What is the difference between standard retrieval and the ColPali approach?"
3. "Why is hybrid search better than vector-only search?"

### Debug Tools

| Tool | Description |
|------|-------------|
| `tools/run_tests.py` | Run the 3 required test questions and log answers |
| `tools/debug_chroma_search.py` | Debug vector search results for a query |
| `tools/debug_retrieve_one.py` | Debug full retrieval pipeline for a single query |
| `tools/inspect_chunk.py` | Inspect a specific chunk by ID |
| `tools/extract_pdf_to_txt.py` | Extract a PDF to text standalone |
| `tools/quick_test.py` | Quick single-question test |

## How It Works

### Data Processing Pipeline (`src/pipeline_ingest.py`)

1. **PDF extraction** (`src/pdf_extract.py`): PyMuPDF opens each PDF and extracts text page-by-page, inserting `[PAGE N]` markers for later chunk identification
2. **Audio transcription** (`src/audio_transcribe.py`): ffmpeg converts MP4 to WAV (16kHz mono), then faster-whisper transcribes with timestamps
3. **Chunking** (`src/chunking.py`): Text is split on paragraph boundaries. PDF text is first split by page markers, then each page is chunked independently. Overlap of 200 chars preserves cross-chunk context
4. **Embedding + storage** (`src/embeddings.py` + `src/vectordb.py`): sentence-transformers encodes chunks to vectors, ChromaDB stores them with source metadata

### Retrieval + Answer Generation (`src/rag.py`)

1. **Vector search**: Query is embedded and searched against ChromaDB (top 50 candidates)
2. **BM25 lexical search**: Custom BM25 implementation searches all stored documents (top 30 candidates)
3. **Merge**: Results from both searches are merged by chunk ID, keeping the better distance score
4. **Rerank**: Combined score = `(0.30 * vector_score) + (1.20 * BM25_score) + source_bias`
   - PDF chunks get +0.80 boost, transcript chunks get -0.40 penalty
5. **Neighbor expansion**: Adjacent PDF pages are added for continuity
6. **Prompt building**: Top-K chunks are formatted with citations `[1], [2]...` and sent to the LLM
7. **Answer generation**: Ollama (or OpenAI) generates the answer using only retrieved context

### Hybrid Retrieval Details

- **Vector search** catches semantic similarity (e.g., "databases for AI" matches "vector stores for GenAI")
- **BM25 search** catches exact terminology (e.g., "ColPali" matches the exact term in documents)
- **Combined** gives better results than either alone - this is why hybrid search is better than vector-only

## File Structure

```
module.4/
├── src/
│   ├── config.py              # Settings from .env
│   ├── pdf_extract.py         # PDF text extraction (PyMuPDF)
│   ├── audio_transcribe.py    # MP4 → WAV → text (ffmpeg + faster-whisper)
│   ├── chunking.py            # Paragraph-aware text chunking
│   ├── embeddings.py          # Embedding generation (sentence-transformers)
│   ├── vectordb.py            # ChromaDB wrapper (persistent storage)
│   ├── llm_clients.py         # Ollama + OpenAI LLM clients
│   ├── rag.py                 # Hybrid retrieval + BM25 + reranking + answer generation
│   └── pipeline_ingest.py     # Full ingestion pipeline entry point
├── tools/
│   ├── run_tests.py           # Run 3 required test questions → logs/run_log.txt
│   ├── debug_chroma_search.py # Debug vector search
│   ├── debug_retrieve_one.py  # Debug full retrieval for one query
│   ├── inspect_chunk.py       # Inspect a chunk by ID
│   ├── extract_pdf_to_txt.py  # Standalone PDF extraction
│   └── quick_test.py          # Quick single-question test
├── data/
│   ├── raw/                   # Source PDFs + MP4s (gitignored)
│   └── processed/             # Extracted text + ChromaDB (gitignored)
├── logs/                      # Test run logs
├── requirements.txt
├── .env.example
└── README.md
```

## Technology Stack

| Component | Technology | Why |
|---|---|---|
| Language | Python 3.10+ | Standard for AI/ML |
| LLM | Ollama (llama3.1) | Free, local, no API keys needed |
| LLM (optional) | OpenAI (gpt-4o-mini) | Higher quality fallback |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Fast, good quality, runs on CPU |
| Vector Store | ChromaDB (persistent) | Simple, Python-native, local storage |
| PDF Extraction | PyMuPDF (fitz) | Fast, reliable page-by-page extraction |
| Audio Transcription | faster-whisper (small model) | Good quality, reasonable speed on CPU |
| Audio Extraction | ffmpeg | Industry standard MP4 → WAV conversion |

## Reflection

### Most Challenging Parts

**Reliable PDF text extraction**: Presentation PDFs don't have clean text flow - text boxes, columns, and graphics create messy extraction. PyMuPDF's page-by-page approach with markers gave the cleanest results, but some slides still had fragmented text that needed careful chunking to remain meaningful.

**Hybrid retrieval tuning**: Getting the right balance between vector and BM25 scores took iteration. Pure vector search missed exact terms (like "ColPali"), pure BM25 missed semantic matches. The final weights (BM25 at 1.20, vector at 0.30) plus source-type bias gave the best results across all three test questions.

**Audio transcription quality**: Lecture recordings with technical terminology produced imperfect transcripts. The solution was to bias retrieval toward PDF sources (which have cleaner text) while still including transcripts for coverage of spoken-only content.

### What I Learned

Building an end-to-end RAG pipeline from scratch - without frameworks like LangChain - revealed how each component contributes to final answer quality. The biggest insight was that retrieval quality matters far more than LLM capability: even a small local model produces good answers when given the right context chunks, while a powerful model struggles with irrelevant context. The hybrid search approach validated why production RAG systems combine multiple retrieval strategies rather than relying on vector search alone.
