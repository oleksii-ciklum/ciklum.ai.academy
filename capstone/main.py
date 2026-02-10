"""CLI entry point for the AI Academy Study Agent."""

import sys
import io
import datetime
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure capstone root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from colorama import init, Fore, Style

init()  # Initialize colorama for Windows

from src.config import SETTINGS
from src.agent.loop import run_agent
from src.agent.tools import search_kb, list_sources, generate_quiz, summarize_topic
from src.agent.reflection import format_reflection

# Logging
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


def log_to_file(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def dim(text: str) -> str:
    return f"{Style.DIM}{text}{Style.RESET_ALL}"


def colored(text: str, color) -> str:
    return f"{color}{text}{Style.RESET_ALL}"


def print_step(step_type: str, content: str):
    """Callback for real-time display of agent thinking."""
    if step_type == "thought":
        print(dim(f"  [Thought] {content}"))
    elif step_type == "action":
        print(dim(f"  [Action]  {content}"))
    elif step_type == "observation":
        print(dim(f"  [Observe] {content[:200]}{'...' if len(content) > 200 else ''}"))
    elif step_type == "answer":
        pass  # Printed separately
    elif step_type == "reflection":
        if "Evaluating" in content:
            print(dim(f"  [Reflect] {content}"))
        else:
            print(colored(f"  [Reflection]\n{content}", Fore.CYAN))


def handle_help():
    print(colored("\n--- Study Agent Commands ---", Fore.YELLOW))
    print("  /help               Show this help message")
    print("  /quiz <topic>       Generate quiz questions on a topic")
    print("  /summary <topic>    Get a structured summary of a topic")
    print("  /sources            List available knowledge base documents")
    print("  /post               Generate a LinkedIn post (agent writes about itself)")
    print("  /eval               Run automated evaluation suite")
    print("  /quit               Exit the agent")
    print("  (anything else)     Ask a question to the study agent")
    print()


def handle_quiz(topic: str):
    if not topic:
        print(colored("Usage: /quiz <topic>", Fore.RED))
        return
    print(dim(f"  Generating quiz on: {topic}..."))
    result = generate_quiz(topic, count=3)
    print(colored(f"\n{result}", Fore.GREEN))
    log_to_file(f"[QUIZ] Topic: {topic}\n{result}\n")


def handle_summary(topic: str):
    if not topic:
        print(colored("Usage: /summary <topic>", Fore.RED))
        return
    print(dim(f"  Summarizing: {topic}..."))
    result = summarize_topic(topic)
    print(colored(f"\n{result}", Fore.GREEN))
    log_to_file(f"[SUMMARY] Topic: {topic}\n{result}\n")


def handle_sources():
    result = list_sources()
    print(colored(f"\n{result}", Fore.GREEN))


def handle_post():
    print(dim("  Agent is generating a LinkedIn post about itself...\n"))

    post_prompt = (
        "Generate a LinkedIn post about yourself. You are an AI Study Agent built for the Ciklum AI Academy. "
        "First, use search_kb to find context about RAG, the academy, and what you can do. "
        "Then use generate_post with that context to create the post. "
        "Return the generated post as your final answer."
    )

    result = run_agent(post_prompt, on_step=print_step)

    # Extract the post text - handle case where answer is a dict
    post_text = result.answer
    if isinstance(post_text, dict):
        post_text = post_text.get("answer", str(post_text))

    # Clean up unicode escape sequences
    post_text = post_text.encode().decode('unicode_escape') if '\\u' in post_text else post_text

    print(colored(f"\n{'='*60}", Fore.YELLOW))
    print(colored("Generated LinkedIn Post:", Fore.GREEN + Style.BRIGHT))
    print(colored(f"{'='*60}", Fore.YELLOW))
    print(post_text)
    print(colored(f"{'='*60}", Fore.YELLOW))

    # Save to file
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    post_path = docs_dir / "linkedin_post.txt"
    with open(post_path, "w", encoding="utf-8") as f:
        f.write(post_text)
    print(colored(f"\nPost saved to: {post_path}", Fore.CYAN))

    if result.reflection:
        print(colored("\nReflection scores:", Fore.CYAN))
        print(format_reflection(result.reflection))

    log_to_file(f"[POST] Generated LinkedIn post\n{post_text}\n")


def handle_eval():
    print(dim("  Running evaluation suite..."))
    try:
        # Import and run evaluation
        from src.evaluation.evaluator import run_evaluation
        run_evaluation()
    except Exception as e:
        print(colored(f"Evaluation error: {e}", Fore.RED))


def handle_question(question: str):
    print()
    log_to_file(f"[Q] {question}")

    result = run_agent(question, on_step=print_step)

    print(colored(f"\n{'='*60}", Fore.YELLOW))
    print(colored("Answer:", Fore.GREEN))
    print(result.answer)
    print(colored(f"{'='*60}", Fore.YELLOW))

    if result.reflection:
        print(colored("\nReflection scores:", Fore.CYAN))
        print(format_reflection(result.reflection))

    if result.retried:
        print(dim("  (Answer was retried due to low quality score)"))

    log_to_file(f"[A] {result.answer}")
    if result.reflection:
        log_to_file(f"[REFLECTION] avg={result.reflection.get('average', '?')}")
    log_to_file("")


def main():
    print(colored("\n" + "="*60, Fore.YELLOW))
    print(colored("  AI Academy Study Agent", Fore.GREEN + Style.BRIGHT))
    print(colored("  Type /help for commands, /quit to exit", Fore.YELLOW))
    print(colored("="*60 + "\n", Fore.YELLOW))

    log_to_file(f"Session started: {datetime.datetime.now().isoformat()}")
    log_to_file(f"LLM: {SETTINGS.llm_provider} / {SETTINGS.ollama_model}\n")

    while True:
        try:
            user_input = input(colored("You: ", Fore.BLUE + Style.BRIGHT)).strip()
        except (EOFError, KeyboardInterrupt):
            print(colored("\nGoodbye!", Fore.YELLOW))
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit" or cmd == "/exit":
                print(colored("Goodbye!", Fore.YELLOW))
                break
            elif cmd == "/help":
                handle_help()
            elif cmd == "/quiz":
                handle_quiz(arg)
            elif cmd == "/summary":
                handle_summary(arg)
            elif cmd == "/sources":
                handle_sources()
            elif cmd == "/post":
                handle_post()
            elif cmd == "/eval":
                handle_eval()
            else:
                print(colored(f"Unknown command: {cmd}. Type /help for commands.", Fore.RED))
        else:
            handle_question(user_input)

    log_to_file(f"Session ended: {datetime.datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
