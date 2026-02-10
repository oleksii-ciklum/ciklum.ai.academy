# Project Reflection - AI Academy Study Agent (Capstone)

## What We Built

An AI-powered Study Agent - a fully agentic RAG system that helps learners study course materials from the Ciklum AI Academy. The agent can answer questions, generate quizzes, summarize topics, and evaluate its own performance, all running locally without any paid API dependencies.

### Core Components Implemented

1. **Data Pipeline**: PDF text extraction (PyMuPDF), lecture audio transcription (faster-whisper), text chunking with overlap, embedding generation (sentence-transformers), persistent storage (ChromaDB).

2. **Hybrid RAG Retrieval**: Combined vector cosine similarity search with a custom BM25 lexical search implementation. Results are merged and reranked using configurable weights, with source-type bias (PDFs boosted over transcripts) and neighbor page expansion for context continuity.

3. **Custom ReAct Agent Loop**: Built from scratch without frameworks (no LangChain, no CrewAI). The agent follows a Think → Act → Observe cycle for up to 5 iterations, selecting from 6 available tools before producing a final answer.

4. **Self-Reflection**: After every answer, the agent evaluates itself on three dimensions - groundedness (is it supported by retrieved context?), relevance (does it address the question?), and completeness (did it cover all aspects?). If the average score falls below a configurable threshold, it retries with a reformulated query.

5. **Tool-Calling System**: Six tools - `search_kb`, `list_sources`, `read_section`, `generate_quiz`, `summarize_topic`, `evaluate_answer` - each with typed arguments and descriptions that the LLM uses to decide what to call.

6. **Automated Evaluation Suite**: Predefined test questions scored via keyword matching, LLM-as-judge, and self-reflection averages. Results logged to file for reproducibility.

---

## What I Learned

- **How agentic systems actually work under the hood**: Building the ReAct loop from scratch forced me to understand every step - prompt construction, output parsing, tool dispatch, iteration control, and error recovery. Using a framework would have hidden all of this.

- **Hybrid retrieval matters**: Pure vector search misses exact terminology; pure keyword search misses semantic meaning. The combination with reranking produced noticeably better results than either alone.

- **Self-reflection is more than a gimmick**: Having the agent evaluate its own answers caught genuinely weak responses. The retry mechanism with query reformulation improved answer quality in measurable ways.

- **Local LLMs are viable but constrained**: Running Llama 3.2 (3B) locally on Ollama proved that you can build a functional agent without API costs, but context window limits and smaller model capacity require careful prompt engineering and chunking strategies.

- **Prompt engineering is iterative**: The system prompt for the ReAct loop went through many iterations. Getting the LLM to consistently follow the THOUGHT/ACTION/OBSERVATION format - especially with a smaller model - required precise formatting and examples.

- **Evaluation is harder than building**: Defining what "good" looks like for an open-ended QA agent is subjective. The three-pronged approach (keywords + LLM judge + self-reflection) provides different perspectives but none is perfect on its own.

---

## Challenges Faced

### 1. Local LLM Output Parsing
**Challenge**: Llama 3.2 (3B) would sometimes break out of the ReAct format, producing malformed JSON tool calls or skipping the THOUGHT step entirely.
**Solution**: Robust regex-based parsing with fallbacks - if structured parsing fails, the agent extracts what it can and continues. Added explicit format reminders in the system prompt.

### 2. Context Window Constraints
**Challenge**: With a 2048–4096 token context window on local models, fitting the system prompt + conversation history + retrieved chunks + tool results was tight.
**Solution**: Aggressive context truncation (`MAX_CTX_CHARS=1800`), keeping only the most recent tool observations, and summarizing rather than dumping raw retrieved text.

### 3. BM25 Implementation from Scratch
**Challenge**: No off-the-shelf BM25 library that worked cleanly with our chunked ChromaDB data.
**Solution**: Implemented BM25 scoring manually with configurable k1 and b parameters, tokenization, and IDF calculation. This was educational but time-consuming.

### 4. Audio Transcription Quality
**Challenge**: Lecture audio transcriptions had lower quality than PDF text - missing punctuation, speaker errors, and topic segmentation issues.
**Solution**: Applied source-type bias in reranking (PDF chunks scored higher than transcript chunks) and added a configurable transcript penalty weight.

### 5. Evaluation Subjectivity
**Challenge**: Hard to define ground truth for open-ended questions about course material.
**Solution**: Combined three complementary scoring methods rather than relying on one. Keyword scoring checks factual coverage, LLM-as-judge assesses holistic quality, and self-reflection measures the agent's own confidence.

### 6. Balancing Simplicity and Depth
**Challenge**: The requirements asked for "simple but functional" while covering all agentic system components.
**Solution**: No external agent frameworks - everything is custom Python. This keeps the codebase understandable while demonstrating genuine understanding of each component.

---

## Technology Stack

| Component | Technology | Why |
|---|---|---|
| Language | Python 3.10+ | Standard for AI/ML |
| LLM | Ollama + Llama 3.2 (3B) | Free, local, no API keys |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Fast, good quality, runs on CPU |
| Vector Store | ChromaDB | Simple, persistent, Python-native |
| PDF Extraction | PyMuPDF (fitz) | Fast, reliable |
| Audio Transcription | faster-whisper (small model) | Good quality, reasonable speed on CPU |
| Agent Framework | Custom (no frameworks) | Demonstrates understanding |

---

## Conclusion

This project covered all major components of an AI-agentic system: data preparation, RAG pipeline design, autonomous reasoning (ReAct), tool calling, self-reflection, and evaluation. The deliberate choice to build everything from scratch - no LangChain, no agent frameworks - meant more work but deeper understanding. The agent is functional, runs locally, and demonstrates the full lifecycle of an agentic AI system.
