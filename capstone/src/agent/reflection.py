"""Self-reflection evaluator for answer quality."""

from typing import Dict, List, Tuple
from ..llm.client import call_llm_json
from .prompts import REFLECTION_PROMPT


def reflect(question: str, answer: str, contexts: List[str]) -> Dict:
    """Evaluate answer quality. Returns dict with scores and reasoning."""
    ctx_text = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts))

    prompt = REFLECTION_PROMPT.format(
        question=question,
        answer=answer,
        contexts=ctx_text,
    )

    result = call_llm_json(prompt)

    # Handle parse failures
    if result.get("_parse_error"):
        return {
            "groundedness": 3,
            "relevance": 3,
            "completeness": 3,
            "reasoning": "Could not parse reflection output. Defaulting to neutral scores.",
            "average": 3.0,
        }

    # Ensure scores are valid integers 0-5
    scores = {}
    for key in ("groundedness", "relevance", "completeness"):
        try:
            val = int(result.get(key, 3))
            scores[key] = max(0, min(5, val))
        except (ValueError, TypeError):
            scores[key] = 3

    scores["reasoning"] = result.get("reasoning", "No reasoning provided.")
    scores["average"] = round(sum(scores[k] for k in ("groundedness", "relevance", "completeness")) / 3, 2)

    return scores


def format_reflection(scores: Dict) -> str:
    """Format reflection scores for display."""
    return (
        f"  Groundedness:  {scores['groundedness']}/5\n"
        f"  Relevance:     {scores['relevance']}/5\n"
        f"  Completeness:  {scores['completeness']}/5\n"
        f"  Average:       {scores['average']}/5\n"
        f"  Reasoning:     {scores['reasoning']}"
    )
