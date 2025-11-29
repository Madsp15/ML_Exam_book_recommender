from autogen import AssistantAgent

from tools.book_api_tool import search_open_library
from utils.utils import FINAL_ANSWER_FORMAT


GENERIC_LIBRARIAN_PROMPT = """
You are a librarian book search assistant using the Open Library API.

BaseFocus:
- {BASE_TOPIC_PROMPT}

Goals:
1. Parse the user's free-form prompt and extract filters: `subject/genre`, `author(s)`, `year_from/year_to`, `max_results`, and key keywords.
2. Prefer subject-first search using `subject` terms; fallback to keyword query when needed.
3. Use available tools efficiently (ignore language filters):
   - Prefer subject-first: search by `subject`.
   - If an author is provided, combine with `author` filtering.
   - Fallback to keyword query if subject results are sparse.
   - Optionally fetch details for promising candidates.
4. Apply filters thoughtfully:
   - Respect `max_results`, avoid duplicates.
   - Prefer recent years when the prompt suggests it; otherwise keep all years.
   - If results are sparse, broaden subject synonyms (e.g., "fantasy" → "Fantasy", "Fantasy fiction", "High fantasy"); do not repeat identical query+filters.
5. Return a final answer that includes for each recommended book:
   - Title, authors, publication year, subjects (if available), and URL.
   - One-line rationale aligned with {BASE_TOPIC_PROMPT}.

Constraints:
- Keep tool calls minimal; respect rate limits.
- Do not invent metadata; only use returned fields.
- Ignore language filters.

Config:
- {YEAR_RANGE}
- {PREFERRED_SUBJECTS}
- {EXCLUDED_SUBJECTS}

{FINAL_ANSWER_FORMAT}
"""


def build_librarian_prompt(
    base_topic_prompt: str,
    year_range: str = "all years",
    preferred_subjects: str = "[]",
    excluded_subjects: str = "[]",
) -> str:
    return GENERIC_LIBRARIAN_PROMPT.format(
        BASE_TOPIC_PROMPT=base_topic_prompt,
        YEAR_RANGE=year_range,
        PREFERRED_SUBJECTS=preferred_subjects,
        EXCLUDED_SUBJECTS=excluded_subjects,
        FINAL_ANSWER_FORMAT=FINAL_ANSWER_FORMAT,
    )


def get_librarian_agent_api_agent(custom_llm_config: dict, base_topic_prompt: str = "General book discovery") -> AssistantAgent:
    system_message = build_librarian_prompt(
        base_topic_prompt=base_topic_prompt,
        year_range="1950–present",
        preferred_subjects="[]",
        excluded_subjects="[]",
    )

    librarian_agent_api_agent = AssistantAgent(
        name="librarianAgentApiAgent",
        llm_config=custom_llm_config,
        system_message=system_message,
    )

    # Register Open Library search tool callable for LLM usage
    librarian_agent_api_agent.register_for_llm(
        name="search_open_library",
        description="Search Open Library for books by query, subject, author, and year range. Returns unified SearchResult list.",
    )(search_open_library)

    return librarian_agent_api_agent
