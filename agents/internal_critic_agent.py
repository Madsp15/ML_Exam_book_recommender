from autogen import AssistantAgent
import json
import logging
import re

from utils.utils import EVALUATION_CRITERIA
from utils.semantic_scorer import calculate_relevance_scores


def get_system_message(terminate_conversation: bool) -> str:
    base_message = f"""
You are an internal critic using a TWO-STAGE evaluation process for book recommendations.

{EVALUATION_CRITERIA}

AUTOMATIC EXCLUSIONS (apply in both stages):
- Drawing/art instruction books, coloring books
- Journals, notebooks, diaries
- Craft/woodwork/technique books
- Books with "No description available" AND empty subjects
- Non-narrative instructional content
- Medical textbooks, pronunciation guides, literary criticism books

NARRATIVE INDICATORS:
- Description tells a story or describes plot/characters
- Subjects include: Fiction, Fantasy fiction, Novels, Stories
- Language: "chronicles", "story of", "adventure", "tale", "follows"

---

STAGE 1 - INITIAL SCREENING (When librarian provides SEARCH RESULTS with titles/snippets):

Your job:
1. Quickly scan titles, snippets, and subjects
2. Identify books that look like narrative fiction matching the request
3. Filter out obvious non-matches (instruction books, non-fiction, etc.)
4. Select 5 most promising books that MIGHT match the requirements

Respond with ONE of:

A) If 5+ books look promising:
```
PROMISING BOOKS (for detailed review):
1. [Book Title] at index #X - Rationale: <why this looks relevant based on title/snippet/subjects>
2. [Book Title] at index #Y - Rationale: <why relevant>
3. [Book Title] at index #Z - Rationale: <why relevant>
4. [Book Title] at index #A - Rationale: <why relevant>
5. [Book Title] at index #B - Score: [B.B] - Rationale: <why relevant>

FETCH DETAILS FOR: #X, #Y, #Z, #A, #B
```

B) If fewer than 5 books look promising:
```
NONE PROMISING

CRITIQUE: <what was requested vs what was found in the search results>

SEARCH SUGGESTIONS FOR LIBRARIAN:
Suggest 2-3 alternative search terms, queries, or author names that could be tried within the available book APIs:
1. Query: "<exact query string>" - Rationale: <why this search term might yield better results>
2. Query: "<alternative keywords>" - Rationale: <why these terms might work better>
3. Author: "<specific author name>" - Rationale: <mention specific books or why this author is relevant>

Note: Only suggest search terms/queries that can be executed within the existing book search APIs. Do not suggest external databases or non-search actions.
```

---

STAGE 1.5 - ERROR RECOVERY (When librarian reports PARTIAL RESULTS or FAILED detail fetches):

Your job:
1. Check how many books were successfully fetched vs failed:
   - SUCCESS: Has title, authors, volumeId, publisher, etc. (description is optional)
   - FAILURE: Returns error message like 'error': 'not_found' or 'error': 'service_unavailable'

2. Count valid fetches:
   - If 3+ books have valid metadata → proceed to STAGE 2
   - If ALL fetches returned error objects → fall back to Stage 1 snippets

3. Handle "No description available":
   - Books with "No description available" are VALID for STAGE 2 evaluation
   - Use other metadata: title, subjects, categories, pageCount, publisher
   - If evaluating these books in STAGE 2 shows they don't match the query well,
     report WEAK MATCHES and suggest new searches

4. Check for MISMATCHES:
   - If librarian reports volume ID/title mismatches, those books are INVALID

5. RECITATION Handling:
   - If librarian says "CONTENT POLICY ISSUE" or mentions copyright detection,
     immediately use Stage 1 snippets


Respond with ONE of:

A) If 3+ successful and VALIDATED detail fetches (no mismatches):
```
ACKNOWLEDGED: Received [N] successful and validated books, proceeding to evaluation.
```
Then immediately evaluate those books as in STAGE 2.

B) If fewer than 3 successful OR all service_unavailable:
```
ACKNOWLEDGED: API detail fetching failed [reason: service unavailable/too many failures]. Using Stage 1 search results with snippets.

OK: Selecting top 3 from initial search results based on titles and snippets.

RESULT:
TOP 3 SELECTED BOOKS:
1. Title: [EXACT title from Stage 1 search results]
   Authors: [EXACT authors from Stage 1]
   Year: [EXACT year from Stage 1]
   URL: [EXACT url from Stage 1]
   Snippet: [relevant snippet from Stage 1]
   Justification: <why this looks most relevant based on snippet>

2. [same format - use EXACT data from original search]

3. [same format - use EXACT data from original search]

TERMINATE.
```
IMPORTANT: Copy titles, authors, years, and URLs EXACTLY as they appeared in the librarian's Stage 1 "SEARCH RESULTS" response.

C) If some successful but WITH MISMATCHES reported by librarian:
```
VALIDATION FAILED: Librarian reported volume ID mismatches. Cannot trust these results.

Fall back to Stage 1 search results with snippets only.
```
Then proceed as in option B above.

D) If fewer than 3 successful AND some books still available:
```
INSUFFICIENT RESULTS: Only [N] books succeeded, need at least 3.

Request the librarian fetch details for the SUGGESTED FALLBACKS from their list.
Or if no fallbacks available: Request alternative books #[next indices from original search].
```

---

STAGE 2 - SCORE AND EVALUATE (When librarian provides DETAILED information with full descriptions):

Your job:
1. Call score_books_by_relevance to calculate semantic relevance scores for the books
   - If the scoring function returns an ERROR (e.g., JSON parsing issues, content policy blocks),
     fall back to evaluating books based on descriptions WITHOUT semantic scores
2. Review the scores alongside full descriptions:
   - Scores ≥80: HIGHLY relevant (excellent semantic match)
   - Scores 65-79: MODERATELY relevant (good match)
   - Scores 50-64: SOMEWHAT relevant (weak match)
   - Scores <50: LOW relevance (poor match)
3. Apply AUTOMATIC EXCLUSIONS rigorously
4. Match descriptions against SPECIFIC task requirements
5. Select the TOP 3 books that ACTUALLY match (not "close enough")

Respond with ONE of:

A) If 3+ books match requirements:
```
[First, call score_books_by_relevance with the user's original prompt and the book details]

OK: <short justification for your top 3 selections, mentioning relevance scores>

RESULT:
TOP 3 SELECTED BOOKS:
1. Title: [title]
   Authors: [authors]
   Year: [year]
   URL: [url]
   Relevance Score: [score from scoring tool]
   Description: [snippet from full description showing relevance]
   Justification: <why this matches requirements based on FULL description and semantic score>
   
2. [same format]

3. [same format]
```

B) If fewer than 3 match:
```
CRITIQUE: <what requirements weren't met in the full descriptions>

SEARCH SUGGESTIONS FOR LIBRARIAN:
<new search strategies with specific queries/authors>
```

---

Rules:
- Stage 1: Quick evaluation - identify 5 promising candidates from titles/snippets
- Stage 2: Deep evaluation - select top 3 actual matches from full descriptions
- Be ABSOLUTELY STRICT - reject books that don't truly match
- Never say "despite X, I selected" - if it doesn't match, REJECT it
- Do NOT invent books - only select from what librarian provides
- If task asks for specific themes (e.g., "about elves"), those themes MUST be explicit in the description
"""
    if terminate_conversation:
        base_message += "\n\n - After providing OK and RESULT in Stage 2, end the conversation with TERMINATE."
    return base_message


def _sanitize_json_string(json_str: str) -> str:
    """
    Sanitize malformed JSON strings from LLM function calls.
    Fixes common issues like invalid escape sequences and Python literals.

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Sanitized JSON string
    """
    # Replace invalid escape sequences - backslash followed by anything except valid escape chars
    # Valid JSON escape sequences: \" \\ \/ \b \f \n \r \t \uXXXX
    json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)

    # Replace Python literals with JSON equivalents
    # Use word boundaries to avoid replacing within strings
    json_str = re.sub(r'\bNone\b', 'null', json_str)
    json_str = re.sub(r'\bTrue\b', 'true', json_str)
    json_str = re.sub(r'\bFalse\b', 'false', json_str)

    return json_str


def score_books_by_relevance(user_prompt: str, books_json: str) -> str:
    """
    Calculate semantic relevance scores for books against user prompt.

    Args:
        user_prompt: Original user query/requirements
        books_json: JSON string containing list of books with title, description, authors, subjects

    Returns:
        Formatted string with books ranked by relevance score (0-100)
    """
    import ast

    # Try multiple parsing strategies
    books = None
    parse_error = None

    # Strategy 1: Direct parsing
    try:
        books = json.loads(books_json)
        logging.info(f"Strategy 1 (direct JSON): Successfully parsed {len(books) if books else 0} books")
    except json.JSONDecodeError as e:
        parse_error = e
        logging.warning(f"Strategy 1 (direct JSON) failed at char {e.pos}: {e.msg}")

        # Strategy 2: Sanitize and retry
        try:
            sanitized_json = _sanitize_json_string(books_json)
            books = json.loads(sanitized_json)
            logging.info(f"Strategy 2 (sanitized JSON): Successfully parsed {len(books) if books else 0} books")
        except json.JSONDecodeError as e2:
            logging.warning(f"Strategy 2 (sanitized JSON) failed at char {e2.pos}: {e2.msg}")

            # Strategy 3: Try ast.literal_eval for Python literal syntax
            try:
                books = ast.literal_eval(books_json)
                logging.info(f"Strategy 3 (ast.literal_eval): Successfully parsed {len(books) if books else 0} books")
            except (ValueError, SyntaxError) as e3:
                logging.error(f"All parsing strategies failed. Original: {parse_error}, Last: {e3}")
                return f"ERROR: Invalid JSON format - Failed to parse book data. Tried 3 strategies. Original error: {str(parse_error)}"

    if not books:
        return "ERROR: No books provided for scoring"

    # Validate books is a list
    if not isinstance(books, list):
        logging.error(f"Expected list of books, got {type(books)}")
        return f"ERROR: Invalid book data format - expected list, got {type(books).__name__}"

    if len(books) == 0:
        return "ERROR: Empty book list provided"

    try:
        logging.info(f"Scoring {len(books)} books for prompt: {user_prompt[:100]}...")

        # Calculate scores
        scored_books = calculate_relevance_scores(user_prompt, books)

        # Format output
        result = [f"SEMANTIC RELEVANCE SCORES (0-100 scale):"]
        result.append("")

        for i, book in enumerate(scored_books, 1):
            score = book.get('relevance_score', 0)
            title = book.get('title', 'Unknown')

            # Categorize relevance with updated thresholds
            if score >= 80:
                label = "HIGHLY relevant"
            elif score >= 65:
                label = "MODERATELY relevant"
            elif score >= 50:
                label = "SOMEWHAT relevant"
            else:
                label = "LOW relevance"

            result.append(f"{i}. Score: {score} - {title}")
            result.append(f"   → {label}")
            result.append("")

        result.append("Note: Scores ≥80 indicate excellent match, 65-79 good match, 50-64 weak match, <50 poor match")

        return "\n".join(result)

    except Exception as e:
        logging.error(f"Error scoring books: {e}")
        return f"ERROR: Failed to score books - {str(e)}"


def get_internal_critic_agent(
    llm_config: dict, terminate_conversation: bool = False
) -> AssistantAgent:
    internal_critic = AssistantAgent(
        name="internal_critic",
        llm_config=llm_config,
        system_message=get_system_message(
            terminate_conversation=terminate_conversation
        ),
    )

    # Register semantic scoring tool
    internal_critic.register_for_llm(
        name="score_books_by_relevance",
        description="""Calculate semantic relevance scores (0-100) for books against the user's original query. 
        Use this in Stage 2 after receiving detailed book information. 
        Parameters:
        - user_prompt (string): The original user query
        - books_json (string): A JSON-formatted string containing a list of book dictionaries. 
          IMPORTANT: Use proper JSON format with null (not None), true/false (not True/False).
        Returns: Books ranked by relevance score with evaluation.""",
    )(score_books_by_relevance)

    return internal_critic
