from datetime import datetime
import json
from pathlib import Path
from typing import List, Literal
import logging

from autogen import ChatResult

logger = logging.getLogger(__name__)


FINAL_ANSWER_FORMAT = """
FINAL ANSWER FORMAT:
When you have fetched details for all books, respond with:
RESULT:
<list of all books found in the format:>
1. Title: <title>
   Authors: <authors>
   Year: <publication year>
   Subjects: <subjects if available>
   URL: <URL>
   Description: <full description from get_book_details>

2. Title: <title>
   Authors: <authors>
   Year: <publication year>
   Subjects: <subjects if available>
   URL: <URL>
   Description: <full description from get_book_details>

... (continue for all books found)

If you cannot find any books with the search criteria, respond with:
JUSTIFICATION:
<brief explanation of why no books were found>
RESULT:
[]
"""

EVALUATION_CRITERIA = """
When evaluating, consider:
    - completeness (1-5): Did the agent satisfy all explicit constraints in the task (e.g., publication year, genre/subject, number of results)?
    - relevance (1-5): Are the returned books relevant to the requested topic and user's needs?
    - accuracy (1-5): Did the agent avoid fabricating book details, and did it explain any limitations of the search results?
    - usefulness (1-5): Is the answer easy to read, with titles, authors, years, subjects, and URLs clearly listed where available?
"""


def _patch_mistral_client():
    """Monkey-patch ConversableAgent to fix 'user after tool' message ordering for Mistral."""
    try:
        from autogen import ConversableAgent

        original_generate_oai_reply = ConversableAgent.generate_oai_reply

        def patched_generate_oai_reply(self, messages=None, sender=None, config=None):
            """Patched generate_oai_reply that transforms messages for Mistral compatibility."""
            if messages:
                transformed = []
                for i, msg in enumerate(messages):
                    if i > 0 and transformed[-1].get("role") == "tool" and msg.get("role") == "user":
                        # Convert user to assistant after tool
                        msg_copy = msg.copy()
                        msg_copy["role"] = "assistant"
                        logger.debug(f"Transformed message role from 'user' to 'assistant' after tool message")
                        transformed.append(msg_copy)
                    else:
                        transformed.append(msg)
                messages = transformed

            return original_generate_oai_reply(self, messages=messages, sender=sender, config=config)

        ConversableAgent.generate_oai_reply = patched_generate_oai_reply
        logger.info("Successfully patched ConversableAgent for message transformation")
    except ImportError as e:
        logger.warning(f"Could not import ConversableAgent - skipping patch: {e}")
    except Exception as e:
        logger.error(f"Error patching ConversableAgent: {e}")


def get_llm_config(
    llm_provider: Literal["mistral", "google", "cerebras"], api_key: str
):
    if llm_provider == "mistral":
        # Apply monkey-patch for Mistral client
        _patch_mistral_client()

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
                    "timeout": 60,
                    "safety_settings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE"
                        }
                    ],
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
