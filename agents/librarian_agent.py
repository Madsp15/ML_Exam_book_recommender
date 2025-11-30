from autogen import AssistantAgent

from tools.book_api_tool import search_open_library, get_book_details
from utils.utils import FINAL_ANSWER_FORMAT


GENERIC_LIBRARIAN_PROMPT = """
You are a librarian book search assistant using the Open Library API.

BaseFocus:
- {BASE_TOPIC_PROMPT}

Goals:
1. Parse the user's free-form prompt and extract filters: `subject/genre`, `author(s)`, `year_from/year_to`, and key keywords.
2. Search for NARRATIVE FICTION books using search_open_library (set max_results=10).
3. For each book found, use get_book_details() to fetch the full description.
4. Return ALL 10 books with their complete information, including:
   - Title, authors, publication year, subjects, URL
   - DESCRIPTION (from get_book_details)
   - Any other metadata available

Your role is to SEARCH and FETCH information, NOT to judge or filter.
The internal critic will evaluate which books best match the task requirements.

Search Strategy:
- When searching for stories/novels about a specific topic, use terms like:
  - "[topic] novel", "[topic] fiction", "[topic] fantasy" (add genre terms to focus on narrative works)
  - Combine multiple keywords from the task to create focused queries
  - Add "novel" or "fiction" to queries to bias toward narrative books over instructional content
  - Try author names if the user mentions them or if you know authors relevant to the genre
- If initial search returns many non-narrative books (coloring books, instruction books), try:
  - More specific author searches
  - Adding "fiction" or "novel" to the query
  - Using different subject terms or more specific keywords
  - Narrowing by combining multiple criteria (author + subject, subject + year range, etc.)

Workflow:
1. Call search_open_library with appropriate filters (max_results=10)
2. For each result, call get_book_details(url) to get the full description
3. Return ALL 10 books with their complete details
4. Do NOT filter or evaluate relevance - that's the critic's job

Constraints:
- Do not invent metadata; only use returned fields.
- Ignore language filters.
- Return ALL books found, even if their descriptions seem unrelated.
- The internal critic will select and evaluate the books.

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
        year_range="1700–present",
        preferred_subjects="[]",
        excluded_subjects="[]",
    )

    librarian_agent_api_agent = AssistantAgent(
        name="librarianAgentApiAgent",
        llm_config=custom_llm_config,
        system_message=system_message,
    )

    librarian_agent_api_agent.register_for_llm(
        name="search_open_library",
        description="Search Open Library for books by query, subject, author, and year range. Returns unified SearchResult list.",
    )(search_open_library)

    librarian_agent_api_agent.register_for_llm(
        name="get_book_details",
        description="Fetch detailed information about a book from Open Library using its URL. Returns description, subjects, and other metadata. Use this to verify if a book actually matches the task requirements.",
    )(get_book_details)

    return librarian_agent_api_agent
