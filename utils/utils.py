from datetime import datetime
import json
from pathlib import Path
from typing import List, Literal

from autogen import ChatResult


FINAL_ANSWER_FORMAT = """
FINAL ANSWER FORMAT:
When you have gathered results satisfying all constraints, respond with:
RESULT:
<list of papers in the format:>
1. Title: <title>
   Authors: <authors>
   Year: <publication year>
   Citations: <number of citations>
   URL: <URL>

If you have found related papers, but some constraints could not be met (e.g., not enough papers with >X citations), respond with:
JUSTIFICATION:
<brief explanation of which constraints could not be met and why>
RESULT:
<list of papers you found>
1. Title: <title>
   Authors: <authors>
   Year: <publication year>
   Citations: <number of citations>
   URL: <URL>

If you cannot find any results meeting the constraints, respond with:
JUSTIFICATION:
<brief explanation of why the constraints could not be met>
RESULT:
[]
"""

EVALUATION_CRITERIA = """
When evaluating, consider:
    - completness (1-5): Did the agent satisfy all explicit constraints in the task (e.g., publication year, citation count, number of results)?
    - relevance (1-5): Are the returned papers relevant to the requested topic?
    - honesty & transparency (1-5): Did the agent avoid fabricating citation counts or details, and did it explain any limitations of the tools used?
    - clarity & structure (1-5): Is the answer easy to read, with titles, authors, years, citation counts, and URLs clearly listed where available?
"""


def get_llm_config(
    llm_provider: Literal["mistral", "google", "cerebras"], api_key: str
):
    if llm_provider == "mistral":
        return {
            "config_list": [
                {
                    "model": "mistral-small-2506",
                    "api_type": "mistral",
                    "api_key": api_key,
                    "api_rate_limit": 0.1,
                    "max_retries": 3,
                    "timeout": 30,
                    "num_predict": -1,
                    "repeat_penalty": 1.1,
                    "stream": False,
                    "seed": 42,
                    "native_tool_calls": False,
                    "cache_seed": None,
                }
            ]
        }
    elif llm_provider == "google":
        return {
            "config_list": [
                {
                    "model": "gemini-2.0-flash",
                    "api_type": "google",
                    "api_key": api_key,
                    "api_rate_limit": 0.1,
                    "max_retries": 3,
                    "num_predict": -1,
                    "repeat_penalty": 1.1,
                    "native_tool_calls": False,
                    "stream": False,
                    "seed": 23,
                    "cache_seed": None,
                    "timeout": 30,
                }
            ]
        }
    elif llm_provider == "cerebras":
        return {
            "config_list": [
                {
                    "model": "llama-3.3-70b",
                    "api_type": "cerebras",
                    "api_key": api_key,
                    "api_rate_limit": 0.1,
                    "max_retries": 3,
                    "num_predict": -1,
                    "repeat_penalty": 1.1,
                    "native_tool_calls": False,
                    "stream": False,
                    "seed": 23,
                    "cache_seed": None,
                    "timeout": 30,
                }
            ]
        }
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")


def get_work_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    p = Path.cwd() / "coding" / timestamp
    p.mkdir(parents=True, exist_ok=True)
    return p


def extract_final_answer(chat: ChatResult, agent_name: str) -> str:
    """
    Extracts the final answer (RESULT:) from the agent's chat history.
    """

    for msg in reversed(chat.chat_history):
        name = msg.get("name", "")
        content = msg.get("content", "")
        if not content or not content.strip():
            continue
        if name != agent_name:
            continue
        if "RESULT:" in content.strip():
            # Extract everything after "RESULT:"
            content = content.replace("TERMINATE:", "")
            result_index = content.index("RESULT:") + len("RESULT:")
            return content[result_index:].strip()
    return "No RESULT found."


def save_results(results: List[dict], filename: str = "latest_results.json"):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    # Prepend timestamp to filename and save under logs/
    base_filename = filename if filename.endswith(".json") else f"{filename}.json"
    log_filename = logs_dir / f"{timestamp}_{base_filename}"

    with open(log_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {log_filename}")
