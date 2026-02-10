"""Automated evaluation suite for the Study Agent."""

import datetime
from pathlib import Path
from typing import List, Dict

from ..agent.loop import run_agent
from ..llm.client import call_llm_json

LOGS_DIR = Path("logs")

# Test suite: questions with expected keywords
TEST_SUITE = [
    {
        "question": "What databases are suitable for GenAI applications?",
        "expected_keywords": ["database", "vector", "genai"],
        "category": "factual",
    },
    {
        "question": "How does chunking strategy affect RAG quality?",
        "expected_keywords": ["chunk", "overlap", "size", "quality", "retrieval"],
        "category": "reasoning",
    },
    {
        "question": "What are the key components of a RAG pipeline?",
        "expected_keywords": ["retrieval", "generation", "embedding", "index"],
        "category": "factual",
    },
]


def keyword_score(answer: str, expected: List[str]) -> float:
    """Score based on fraction of expected keywords present in answer."""
    if not answer or not expected:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected if kw.lower() in answer_lower)
    return round(found / len(expected), 2)


def llm_judge(question: str, answer: str) -> Dict:
    """Use LLM-as-judge to evaluate answer quality."""
    prompt = f"""Rate this answer on a scale of 1-5 for accuracy and helpfulness.

Question: {question}
Answer: {answer}

Respond with JSON only:
{{"score": <int 1-5>, "reasoning": "brief explanation"}}"""

    result = call_llm_json(prompt)
    if result.get("_parse_error"):
        return {"score": 3, "reasoning": "Could not parse LLM judge output."}

    try:
        score = max(1, min(5, int(result.get("score", 3))))
    except (ValueError, TypeError):
        score = 3

    return {"score": score, "reasoning": result.get("reasoning", "")}


def run_evaluation():
    """Run the full evaluation suite and log results."""
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / "eval_results.txt"

    results = []
    total_keyword = 0.0
    total_llm = 0.0
    total_reflection = 0.0

    print(f"\nRunning {len(TEST_SUITE)} evaluation questions...\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Run: {datetime.datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        completed = 0
        for i, test in enumerate(TEST_SUITE, 1):
            q = test["question"]
            print(f"  [{i}/{len(TEST_SUITE)}] {q}")

            try:
                # Run agent
                agent_result = run_agent(q)
                answer = agent_result.answer
            except Exception as e:
                print(f"    SKIPPED (error: {e})")
                f.write(f"Q{i}: {q}\n")
                f.write(f"SKIPPED due to error: {e}\n")
                f.write("-" * 40 + "\n\n")
                continue

            # Keyword scoring
            kw_score = keyword_score(answer, test["expected_keywords"])
            total_keyword += kw_score

            # LLM judge
            try:
                judge = llm_judge(q, answer)
            except Exception:
                judge = {"score": 0, "reasoning": "LLM judge timed out"}
            total_llm += judge["score"]

            # Reflection score
            ref_avg = agent_result.reflection["average"] if agent_result.reflection else 0.0
            total_reflection += ref_avg

            # Count tool calls
            tool_calls = [s for s in agent_result.thinking_log if s["type"] == "action"]
            completed += 1

            result = {
                "question": q,
                "category": test["category"],
                "answer": answer[:300],
                "keyword_score": kw_score,
                "llm_judge_score": judge["score"],
                "llm_judge_reasoning": judge["reasoning"],
                "reflection_avg": ref_avg,
                "tool_calls": len(tool_calls),
                "retried": agent_result.retried,
            }
            results.append(result)

            # Log to file
            f.write(f"Q{i}: {q}\n")
            f.write(f"Category: {test['category']}\n")
            f.write(f"Answer: {answer[:500]}\n")
            f.write(f"Keyword Score: {kw_score}\n")
            f.write(f"LLM Judge: {judge['score']}/5 - {judge['reasoning']}\n")
            f.write(f"Reflection: {ref_avg}/5\n")
            f.write(f"Tool Calls: {len(tool_calls)}\n")
            f.write(f"Retried: {agent_result.retried}\n")
            f.write("-" * 40 + "\n\n")

            print(f"    Keyword: {kw_score} | LLM Judge: {judge['score']}/5 | Reflection: {ref_avg}/5")

        n = completed or 1
        summary = (
            f"\n{'='*70}\n"
            f"SUMMARY ({completed}/{len(TEST_SUITE)} questions completed)\n"
            f"  Avg Keyword Score:    {total_keyword/n:.2f}\n"
            f"  Avg LLM Judge Score:  {total_llm/n:.2f}/5\n"
            f"  Avg Reflection Score: {total_reflection/n:.2f}/5\n"
            f"{'='*70}\n"
        )

        f.write(summary)
        print(summary)
        print(f"Results saved to: {log_path}")
