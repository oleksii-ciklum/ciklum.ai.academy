"""Prompt templates for the ReAct agent."""

from .tools import TOOL_DESCRIPTIONS


def _format_tool_descriptions() -> str:
    lines = []
    for name, info in TOOL_DESCRIPTIONS.items():
        args_str = ", ".join(f"{k}: {v}" for k, v in info["args"].items()) if info["args"] else "none"
        lines.append(f"  - {name}({args_str}): {info['description']}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are a Study Agent for the Ciklum AI Academy. You help students learn by searching a knowledge base, generating quizzes, creating summaries, and answering questions.

You have access to these tools:
{_format_tool_descriptions()}

## How to respond

You MUST respond with valid JSON in one of these two formats:

### To use a tool:
{{"thought": "your reasoning about what to do next", "action": "tool_name", "args": {{"arg1": "value1"}}}}

### To give a final answer:
{{"thought": "your reasoning", "answer": "your complete answer to the user"}}

## Rules:
1. Think step by step about what information you need.
2. Use search_kb to find relevant information before answering.
3. You can call multiple tools across iterations to gather more info.
4. Include citations like [1], [2] in your final answer when referencing retrieved content.
5. If the knowledge base doesn't have relevant info, say so honestly.
6. Always respond with valid JSON only. No other text before or after the JSON."""


REFLECTION_PROMPT = """Evaluate the following answer for quality. Be strict and honest.

Question: {question}

Answer: {answer}

Retrieved contexts used:
{contexts}

Score each dimension from 0 to 5:
1. Groundedness: Is the answer supported by the retrieved context? (0=fabricated, 5=fully grounded)
2. Relevance: Does the answer address the question asked? (0=off-topic, 5=directly answers)
3. Completeness: Does the answer cover all aspects? (0=misses everything, 5=comprehensive)

You MUST respond with ONLY valid JSON in this exact format:
{{"groundedness": <int 0-5>, "relevance": <int 0-5>, "completeness": <int 0-5>, "reasoning": "brief explanation of scores"}}"""


RETRY_PROMPT = """The previous answer scored poorly on quality evaluation.

Original question: {question}
Previous answer quality issues: {reasoning}

Please reformulate and search again to provide a better answer. Think about what information might have been missed."""
