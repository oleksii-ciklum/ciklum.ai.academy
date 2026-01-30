from src.rag import RAGBot

b = RAGBot()
q = "What are the production dos for RAG?"
print(b.answer(q))
