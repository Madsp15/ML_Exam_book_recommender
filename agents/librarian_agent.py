from autogen import AssistantAgent
import logging

from tools import search_google_books, get_book_details_google
from utils.utils import FINAL_ANSWER_FORMAT
from utils.semantic_scorer import calculate_relevance_scores


GENERIC_LIBRARIAN_PROMPT = """
You are a librarian book search assistant using the Google Books API.

BaseFocus:
- {BASE_TOPIC_PROMPT}

TWO-STAGE SEARCH WORKFLOW:

STAGE 1 - Initial Search (Always do this first):
1. Parse the user's request and extract filters: subject/genre, author, year range, keywords.
2. Call search_google_books with max_results=10 (ALWAYS 10, even if user asks for "top 3").
3. Return the search results AS-IS with basic info: title, authors, year, subjects, snippet, URL.
4. DO NOT call get_book_details_google yet. Wait for critic to evaluate the search results.
5. Format results in a numbered list so critic can reference them easily.

STAGE 2 - Detailed Fetching (Only after critic identifies promising books):
1. The critic will tell you which books look promising (e.g., "FETCH DETAILS FOR: #1, #3, #5, #7, #9").
2. Extract volume IDs from the URLs of those specific books.
3. Call get_book_details_google(volume_id) for the 5 most promising books the critic identified.
4. Return the detailed information with full descriptions.
5. The critic will then do final evaluation with complete information.

IMPORTANT - Volume ID Extraction:
- Google Books URLs: http://books.google.com/books?id=VOLUME_ID&...
- Extract ID: url.split('id=')[1].split('&')[0]
- The get_book_details_google function already handles URL parsing

ERROR HANDLING (CRITICAL):
When detail fetching fails (returns {{'error': ...}}):
1. **Track Volume ID Mappings**: When reporting results, cross-reference volume IDs from Stage 1 with returned details
   - If volume ID in response doesn't match requested ID -> Report MISMATCH
   - If title in detailed response doesn't match title from Stage 1 search -> Report MISMATCH
   
2. **Partial Results Handling**:
   - Skip failed books and report which ones failed and why
   - If you have successful results, return those with a note about failures
   - VALIDATE: Check if returned book title matches expected title from Stage 1
   
3. **Complete Failure Handling**:
   - If ALL fetches fail with 404/503/timeouts:
     * STOP trying more volume IDs - the API has issues
     * Suggest using ORIGINAL SEARCH RESULTS (Stage 1 snippets) instead
     * Format: "ALL detail fetches failed. Recommend using Stage 1 search results with snippets."
   
4. **Format Response**:
   ```
   PARTIAL RESULTS (X successful, Y failed):
   
   SUCCESSFUL:
   [List successful book details here with validation note]
   
   FAILED:
   - Book #N (Title): Failed because [error_type: not_found/timeout/service_unavailable/etc]
   - Book #M (Title): Failed because [error_type]
   
   MISMATCHES (if any):
   - Volume ID [ID] expected 'Title A' but got 'Title B'
   
   JUSTIFICATION: [Why partial results or next steps]
   
   SUGGESTED FALLBACKS: 
   - Try fetching details for #[alternative indices from original search]
   - OR if all failed/circuit breaker: "Recommend using Stage 1 search results with snippets only"
   ```
   
5. NEVER return empty "RESULT: []" - always provide context about what happened
6. LIMIT: If 2 consecutive detail fetch rounds all fail, STOP and recommend using Stage 1 results

Search Strategy:
- Initial search: Use "[topic] novel", "[topic] fiction", or "[topic] + subject:genre"
- ALWAYS use max_results=10 (even if user wants only 3 final results)
- Use subject parameter for genre filtering (e.g., subject="fantasy")
- Use author parameter when searching specific authors

Responding to Critic Feedback:
- If critic says "NONE PROMISING" with SEARCH SUGGESTIONS → Try the suggested alternative search
- If critic says "FETCH DETAILS FOR: [list]" → Extract volume IDs and call get_book_details_google for those books (max 5)
- If some detail fetches fail → Report partial results and suggest fallback options
- If critic approves in Stage 2 → Job done!

Example Flow:
User asks: "Find fantasy books about elves"

Your Stage 1 Response:
"SEARCH RESULTS (10 books):

1. Title: The Elvish Chronicles
   Authors: Jane Smith
   Year: 2015
   Subjects: Fiction, Fantasy, Elves
   Snippet: A tale of ancient elves in Middle Kingdom...
   URL: http://books.google.com/books?id=ABC123&
   
2. Title: Forest of the Fae
   Authors: John Doe
   Year: 2018
   Subjects: Fiction, Fantasy
   Snippet: Elven warriors defend their homeland...
   URL: http://books.google.com/books?id=XYZ789&

[... continue for all 10 results ...]"

Critic Response Example:
"PROMISING BOOKS: #1, #2, #7, #8, #9 look relevant. FETCH DETAILS FOR: #1, #2, #7, #8, #9"

Your Stage 2 Response:
[Call get_book_details_google for those 5 books and return full descriptions]

Constraints:
- Maximum 1 search per turn
- Maximum 5 detail fetches per turn (to avoid API limits and context overflow)
- Do not filter or evaluate - that's the critic's job
- In Stage 1, just return the search results, don't fetch details yet

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


# Store the current user query for semantic scoring
_current_user_query = ""


def set_current_query(query: str):
    """Store the current user query for semantic scoring"""
    global _current_user_query
    _current_user_query = query
    logging.info(f"Set current query for semantic scoring: {query[:100]}...")


def search_google_books_with_scoring(
    query: str,
    year_from: int = None,
    year_to: int = None,
    max_results: int = 10,
    author: str = None,
    subject: str = None,
):
    """
    Wrapper for search_google_books that automatically calculates semantic relevance scores.
    Returns SearchResult objects with added relevance_score attribute.
    """
    # Call original search function
    results = search_google_books(
        query=query,
        year_from=year_from,
        year_to=year_to,
        max_results=max_results,
        author=author,
        subject=subject,
    )

    if not results:
        logging.warning("No results from Google Books search")
        return results

    # Get user query from context (use search query as fallback)
    user_query = _current_user_query if _current_user_query else query

    # Convert SearchResult objects to dicts for scoring
    books_as_dicts = []
    for result in results:
        book_dict = {
            "title": result.title,
            "description": result.snippet,
            "snippet": result.snippet,
            "authors": result.authors,
            "subjects": result.subjects or [],
            "publication_year": result.publication_year,
            "url": result.url,
            "isbn": result.isbn,
            "volume_id": result.volume_id,
        }
        books_as_dicts.append(book_dict)

    # Calculate semantic scores
    try:
        logging.info(f"Calculating semantic scores for {len(books_as_dicts)} books...")
        scored_books = calculate_relevance_scores(user_query, books_as_dicts)

        # Add scores back to SearchResult objects
        for i, result in enumerate(results):
            if i < len(scored_books):
                result.relevance_score = scored_books[i].get("relevance_score", 0)

        # Sort by relevance score (highest first)
        results.sort(key=lambda x: getattr(x, 'relevance_score', 0), reverse=True)

        logging.info(f"Top 3 relevance scores: {[getattr(r, 'relevance_score', 0) for r in results[:3]]}")

    except Exception as e:
        logging.error(f"Failed to calculate semantic scores: {e}")
        # Return results without scores if scoring fails
        for result in results:
            result.relevance_score = None

    return results


def get_librarian_agent_api_agent(custom_llm_config: dict, base_topic_prompt: str = "General book discovery") -> AssistantAgent:
    system_message = build_librarian_prompt(
        base_topic_prompt=base_topic_prompt,
        year_range="all years",
        preferred_subjects="[]",
        excluded_subjects="[]",
    )

    librarian_agent_api_agent = AssistantAgent(
        name="librarianAgentApiAgent",
        llm_config=custom_llm_config,
        system_message=system_message,
    )

    librarian_agent_api_agent.register_for_llm(
        name="search_google_books",
        description="Search Google books for books by query, subject, author, and year range. Returns unified SearchResult list.",
    )(search_google_books)

    librarian_agent_api_agent.register_for_llm(
        name="get_book_details_google",
        description="Fetch detailed information about a book from Google Books using its volume ID.",
    )(get_book_details_google)

    return librarian_agent_api_agent
