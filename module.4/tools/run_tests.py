from pathlib import Path
from src.rag import RAGBot

QUESTIONS = [
    "What are the production 'Do's' for RAG?",
    "What is the difference between standard retrieval and the ColPali approach?",
    "Why is hybrid search better than vector-only search?",
]

def main():
    bot = RAGBot()
    lines = []
    for q in QUESTIONS:
        lines.append("=" * 80)
        lines.append("Q: " + q)
        lines.append("A: " + bot.answer(q))

    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("logs/run_log.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote logs/run_log.txt")

if __name__ == "__main__":
    main()
