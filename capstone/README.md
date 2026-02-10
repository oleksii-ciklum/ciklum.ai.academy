# Capstone: AI Academy Study Agent

An intelligent study assistant built with a custom ReAct agent loop, hybrid RAG retrieval, and self-reflection. The agent reasons about what to retrieve, takes multiple tool-based actions to compile answers, evaluates its own quality, and offers study features like quizzes and summaries.

## Architecture

```
User (CLI) --> Agent Loop (ReAct) --> Tools --> LLM (Ollama) --> Reflection --> Final Answer
                                       |
                       +---------------+---------------+
                       |               |               |
                 search_kb      generate_quiz    summarize_topic
                 list_sources   read_section     evaluate_answer
```

See `architecture.mmd` for the full Mermaid diagram.

### Key Design Decisions

- **Custom agent loop** (no frameworks): demonstrates understanding of ReAct pattern
- **Hybrid retrieval**: vector (ChromaDB) + BM25 lexical search with reranking
- **Self-reflection**: LLM evaluates its own answers for groundedness, relevance, completeness
- **Ollama (local LLM)**: free, private, no API keys needed
- **Builds on Module 4 RAG**: reuses chunking, embeddings, vector store, and retrieval logic

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally
- ffmpeg (for audio extraction from MP4)

### Installation

```bash
cd capstone
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env if needed (defaults work with local Ollama)
```

Pull the Ollama model:
```bash
ollama pull llama3.2:3b
```

### Data Preparation

Place these files in `data/raw/`:
- `rag_intro.pdf`
- `databases_for_genai.pdf`
- `rag_intro.mp4`
- `databases_for_genai.mp4`

### Ingestion

```bash
python ingest.py
```

This extracts PDFs, transcribes audio, chunks text, generates embeddings, and stores everything in ChromaDB.

## Usage

### Interactive Chat

```bash
python main.py
```

### Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/quiz <topic>` | Generate quiz questions on a topic |
| `/summary <topic>` | Get a structured summary of a topic |
| `/sources` | List knowledge base documents |
| `/post` | Agent generates a LinkedIn post about itself |
| `/eval` | Run automated evaluation suite |
| `/quit` | Exit the agent |

Any other input is treated as a question for the study agent.

### Evaluation

```bash
python evaluate.py
```

Runs 3 predefined test questions and scores answers using keyword matching, LLM-as-judge, and self-reflection. Results are saved to `logs/eval_results.txt`.

## How It Works

### ReAct Agent Loop

1. User asks a question
2. Agent **THINKS**: decides which tool(s) to call
3. Agent **ACTS**: calls the selected tool
4. Agent **OBSERVES**: processes tool results
5. Repeat steps 2-4 until enough info gathered (max 5 iterations)
6. Agent produces **FINAL ANSWER** with citations
7. Agent **REFLECTS**: evaluates answer quality (groundedness, completeness, relevance)
8. If reflection score < threshold: retry with reformulated query (max 1 retry)
9. Display final answer + reflection scores

### Tools

| Tool | Description |
|------|-------------|
| `search_kb` | Hybrid RAG search over the knowledge base |
| `list_sources` | Lists available documents |
| `read_section` | Reads a specific page from a PDF |
| `generate_quiz` | Creates multiple-choice quiz questions |
| `summarize_topic` | Generates structured topic summaries |
| `evaluate_answer` | Self-evaluates answer quality |
| `generate_post` | Generates a LinkedIn post about the agent |

### Hybrid Retrieval

- **Vector search**: ChromaDB cosine similarity (sentence-transformers)
- **Lexical search**: Custom BM25 implementation
- **Reranking**: weighted combination of vector + BM25 scores + source bias (PDF preferred over transcript)
- **Neighbor expansion**: adds adjacent PDF pages for context continuity

## File Structure

```
capstone/
├── src/
│   ├── config.py              # Settings from .env
│   ├── rag/
│   │   ├── chunking.py        # Text chunking
│   │   ├── embeddings.py      # Embedding generation
│   │   ├── vectordb.py        # ChromaDB wrapper
│   │   ├── bm25.py            # BM25 lexical search
│   │   ├── retrieval.py       # Hybrid search + reranking
│   │   ├── pdf_extract.py     # PDF text extraction
│   │   └── audio_transcribe.py# Audio transcription
│   ├── agent/
│   │   ├── loop.py            # ReAct agent loop
│   │   ├── tools.py           # Tool definitions
│   │   ├── reflection.py      # Self-reflection evaluator
│   │   └── prompts.py         # Prompt templates
│   ├── llm/
│   │   └── client.py          # Ollama + OpenAI client
│   └── evaluation/
│       └── evaluator.py       # Automated test suite
├── data/
│   ├── raw/                   # Source PDFs + MP4s (gitignored)
│   └── processed/             # Extracted text + ChromaDB
├── logs/                      # Conversation + evaluation logs
├── main.py                    # CLI entry point
├── ingest.py                  # Data ingestion entry point
├── evaluate.py                # Evaluation entry point
├── requirements.txt
├── .env.example
└── architecture.mmd           # Mermaid architecture diagram
```
