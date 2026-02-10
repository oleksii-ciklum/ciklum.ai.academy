"""Agent tool definitions and implementations."""

from typing import Dict, Any, List, Callable, Optional
from ..rag.retrieval import Retriever, clean_context_text, _extract_relevant_window, MAX_CTX_CHARS
from ..llm.client import call_llm

# Lazy-initialized retriever (shared across tools)
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# ---- Tool implementations ----

def search_kb(query: str) -> str:
    """Hybrid RAG search, returns top-k relevant chunks with sources."""
    retriever = get_retriever()
    results = retriever.retrieve(query)
    if not results:
        return "No relevant results found in the knowledge base."

    parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {}) or {}
        src = meta.get("source", "unknown")
        snippet = _extract_relevant_window(query, r.get("text", ""), MAX_CTX_CHARS)
        snippet = clean_context_text(snippet)
        parts.append(f"[{i}] Source: {src}\n{snippet[:800]}")

    return "\n\n".join(parts)


def list_sources() -> str:
    """Lists available knowledge base documents."""
    retriever = get_retriever()
    sources = retriever.get_sources()
    if not sources:
        return "No sources found in the knowledge base."
    lines = ["Available knowledge base sources:"]
    for s in sources:
        lines.append(f"  - {s}")
    return "\n".join(lines)


def read_section(source: str, page: int) -> str:
    """Reads a specific page from a PDF source."""
    retriever = get_retriever()
    text = retriever.get_page_text(source, page)
    return clean_context_text(text)[:2000]


def generate_quiz(topic: str, count: int = 3) -> str:
    """Generates N quiz questions on a topic from retrieved content."""
    retriever = get_retriever()
    results = retriever.retrieve(topic, top_k=5)

    context = "\n\n".join(
        clean_context_text(r.get("text", ""))[:600]
        for r in results
    )

    prompt = f"""Based on the following study material, generate exactly {count} multiple-choice quiz questions about "{topic}".

For each question:
- Write the question
- Provide 4 options (A, B, C, D)
- Mark the correct answer
- Add a brief explanation

Study material:
{context}

Format each question as:
Q1: [question]
A) [option]
B) [option]
C) [option]
D) [option]
Correct: [letter]
Explanation: [brief explanation]
"""

    return call_llm(prompt, system="You are a helpful quiz generator for students. Create clear, educational quiz questions.")


def summarize_topic(topic: str) -> str:
    """Generates a structured summary of a topic from the knowledge base."""
    retriever = get_retriever()
    results = retriever.retrieve(topic, top_k=7)

    if not results:
        return f"No content found for topic: {topic}"

    context = retriever.format_contexts(topic, results)

    prompt = f"""Create a structured summary of "{topic}" based only on the context below.

Format:
## {topic}

### Key Concepts
- [bullet points]

### Details
[2-3 paragraphs]

### Key Takeaways
- [bullet points]

Context:
{context}

Summary:"""

    return call_llm(prompt, system="You are a study assistant. Summarize accurately using only the provided context. Add citations like [1], [2].")


def generate_post(context: str, platform: str = "LinkedIn") -> str:
    """Generates a professional social media post about this agent for the given platform."""
    prompt = f"""Write a short {platform} post (5-7 sentences) about an AI Study Agent.

The post must:
- Be written in the first person as the agent itself ("I am...")
- Explain what the agent does and how it was built
- Mention it was created as part of the Ciklum AI Academy
- Include a mention of @Ciklum
- Be professional, authentic, and concise
- End with 3-5 relevant hashtags

Use the following context about the agent and the academy to make it specific and accurate:
{context}

Write the post now:"""

    return call_llm(prompt, system="You are a professional social media content writer. Write concise, engaging posts. Do not use emojis excessively.")


def evaluate_answer(question: str, answer: str, contexts: List[str]) -> str:
    """Self-evaluates answer quality, returns score + reasoning."""
    ctx_text = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts))

    prompt = f"""Evaluate the following answer for quality. Score each dimension 0-5.

Question: {question}

Answer: {answer}

Retrieved Context:
{ctx_text}

Score these dimensions:
1. Groundedness (0-5): Is the answer supported by the retrieved context?
2. Relevance (0-5): Does the answer actually address the question?
3. Completeness (0-5): Does the answer cover all aspects of the question?

Respond in this exact format:
Groundedness: [score]/5 - [brief reason]
Relevance: [score]/5 - [brief reason]
Completeness: [score]/5 - [brief reason]
Average: [average]/5
"""

    return call_llm(prompt, system="You are a strict answer quality evaluator. Be honest and critical.")


# ---- Tool registry ----

TOOL_DESCRIPTIONS = {
    "search_kb": {
        "description": "Hybrid RAG search over the knowledge base. Returns top relevant chunks with source citations.",
        "args": {"query": "str - the search query"},
    },
    "list_sources": {
        "description": "Lists all available documents in the knowledge base.",
        "args": {},
    },
    "read_section": {
        "description": "Reads a specific page from a PDF source document.",
        "args": {"source": "str - source filename (e.g. 'rag_intro.pdf.txt')", "page": "int - page number"},
    },
    "generate_quiz": {
        "description": "Generates multiple-choice quiz questions on a topic from the knowledge base.",
        "args": {"topic": "str - the topic for quiz questions", "count": "int - number of questions (default 3)"},
    },
    "summarize_topic": {
        "description": "Generates a structured summary of a topic using knowledge base content.",
        "args": {"topic": "str - the topic to summarize"},
    },
    "evaluate_answer": {
        "description": "Evaluates an answer's quality (groundedness, relevance, completeness) against retrieved context.",
        "args": {"question": "str", "answer": "str", "contexts": "list[str]"},
    },
    "generate_post": {
        "description": "Generates a professional LinkedIn post about this agent using gathered context. Call search_kb first to gather context, then pass it here.",
        "args": {"context": "str - relevant context about the agent and academy", "platform": "str - social media platform (default: LinkedIn)"},
    },
}

TOOL_FUNCTIONS: Dict[str, Callable] = {
    "search_kb": lambda args: search_kb(args["query"]),
    "list_sources": lambda args: list_sources(),
    "read_section": lambda args: read_section(args["source"], int(args["page"])),
    "generate_quiz": lambda args: generate_quiz(args["topic"], int(args.get("count", 3))),
    "summarize_topic": lambda args: summarize_topic(args["topic"]),
    "evaluate_answer": lambda args: evaluate_answer(args["question"], args["answer"], args["contexts"]),
    "generate_post": lambda args: generate_post(args["context"], args.get("platform", "LinkedIn")),
}


def dispatch_tool(tool_name: str, args: Dict[str, Any]) -> str:
    """Execute a tool by name with the given arguments."""
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        return f"Unknown tool: {tool_name}. Available tools: {', '.join(TOOL_FUNCTIONS.keys())}"
    try:
        return fn(args)
    except Exception as e:
        return f"Tool '{tool_name}' failed: {e}"
