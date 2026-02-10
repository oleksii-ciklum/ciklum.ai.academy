"""ReAct agent loop implementation."""

import json
from typing import List, Dict, Any, Optional, Callable
from ..config import SETTINGS
from ..llm.client import call_llm_json
from .prompts import SYSTEM_PROMPT, RETRY_PROMPT
from .tools import dispatch_tool
from .reflection import reflect, format_reflection


class AgentResult:
    """Container for agent execution results."""
    def __init__(self):
        self.answer: str = ""
        self.thinking_log: List[Dict[str, str]] = []  # list of {type, content}
        self.reflection: Optional[Dict] = None
        self.retried: bool = False
        self.contexts_used: List[str] = []


def _build_messages(system: str, conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build the message list for the LLM."""
    messages = [{"role": "system", "content": system}]
    messages.extend(conversation)
    return messages


def run_agent(question: str, on_step: Optional[Callable[[str, str], None]] = None) -> AgentResult:
    """Run the ReAct agent loop for a question.

    Args:
        question: The user's question.
        on_step: Optional callback(step_type, content) for real-time display.
                 step_type is one of: "thought", "action", "observation", "answer", "reflection"
    """
    result = AgentResult()

    def log(step_type: str, content: str):
        result.thinking_log.append({"type": step_type, "content": content})
        if on_step:
            on_step(step_type, content)

    def _run_loop(user_message: str) -> str:
        """Execute one full agent loop, return the final answer."""
        conversation = [{"role": "user", "content": user_message}]
        contexts_collected: List[str] = []

        for iteration in range(SETTINGS.max_agent_iterations):
            messages = _build_messages(SYSTEM_PROMPT, conversation)
            response = call_llm_json("", messages=messages)

            # Handle parse errors
            if response.get("_parse_error"):
                raw = response.get("_raw", "")
                log("thought", f"(LLM returned non-JSON, treating as direct answer)")
                return raw

            thought = response.get("thought", "")
            if thought:
                log("thought", thought)

            # Check for final answer
            if "answer" in response and response["answer"]:
                answer = response["answer"]
                # Handle case where answer is wrapped in another dict
                if isinstance(answer, dict):
                    answer = answer.get("answer", str(answer))
                # Ensure answer is a string
                answer = str(answer) if not isinstance(answer, str) else answer
                log("answer", answer)
                result.contexts_used = contexts_collected
                return answer

            # Check for tool action
            action = response.get("action", "")
            args = response.get("args", {})

            if not action:
                # No action and no answer - treat thought as answer
                if thought:
                    log("answer", thought)
                    return thought
                return "I wasn't able to formulate a response. Please try rephrasing your question."

            log("action", f"{action}({args})")

            # Execute tool
            tool_result = dispatch_tool(action, args if isinstance(args, dict) else {})
            log("observation", tool_result[:500] + ("..." if len(tool_result) > 500 else ""))

            # Track contexts from search
            if action == "search_kb" and tool_result and "No relevant results" not in tool_result:
                contexts_collected.append(tool_result[:1000])

            # Feed result back into conversation
            assistant_msg = json.dumps({"thought": thought, "action": action, "args": args})
            conversation.append({"role": "assistant", "content": assistant_msg})
            conversation.append({"role": "user", "content": f"Tool result from {action}:\n{tool_result}"})

        # Max iterations reached
        log("thought", "Maximum iterations reached. Providing best answer from gathered information.")
        # Try one more time to get a final answer
        conversation.append({"role": "user", "content": "You have reached the maximum number of tool calls. Please provide your final answer now based on what you have gathered. Respond with: {\"thought\": \"...\", \"answer\": \"...\"}"})
        messages = _build_messages(SYSTEM_PROMPT, conversation)
        response = call_llm_json("", messages=messages)

        answer = response.get("answer", response.get("thought", "Unable to complete the response within the allowed iterations."))
        result.contexts_used = contexts_collected
        return answer

    # First run
    answer = _run_loop(question)
    result.answer = answer

    # Reflection
    if result.contexts_used:
        log("reflection", "Evaluating answer quality...")
        scores = reflect(question, answer, result.contexts_used)
        result.reflection = scores
        log("reflection", format_reflection(scores))

        # Retry if quality is low
        if scores["average"] < SETTINGS.reflection_threshold and SETTINGS.max_retries > 0:
            log("thought", f"Quality score {scores['average']}/5 is below threshold {SETTINGS.reflection_threshold}. Retrying...")
            result.retried = True

            retry_message = RETRY_PROMPT.format(
                question=question,
                reasoning=scores["reasoning"],
            )
            answer2 = _run_loop(retry_message + "\n\nOriginal question: " + question)

            # Re-evaluate
            if result.contexts_used:
                scores2 = reflect(question, answer2, result.contexts_used)
                log("reflection", "Retry reflection:\n" + format_reflection(scores2))

                # Use the better answer
                if scores2["average"] >= scores["average"]:
                    result.answer = answer2
                    result.reflection = scores2
                else:
                    log("thought", "Retry did not improve quality. Keeping original answer.")
    else:
        # No contexts = likely a tool listing or direct response, no reflection needed
        result.reflection = None

    return result
