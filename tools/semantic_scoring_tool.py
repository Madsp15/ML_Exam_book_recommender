"""
Semantic scoring tool for the librarian agent.
Provides a function to score and rank books based on semantic similarity to user prompt.
"""
import logging
from typing import List, Dict, Any
from utils.semantic_scorer import get_scorer


def score_books_by_relevance(user_prompt: str, books_json: str) -> str:
    """
    Score and rank books by semantic relevance to the user's prompt.

    This tool uses AI embeddings to calculate how well each book matches
    the user's requirements beyond simple keyword matching.

    Args:
        user_prompt: The user's original search query/requirements
        books_json: JSON string containing list of books with fields:
                   title, description/snippet, authors, subjects, etc.

    Returns:
        Formatted string with books ranked by relevance score (0-100)

    Example:
        Input books: [{"title": "Book A", "snippet": "...", ...}, ...]
        Output:
        SCORED BOOKS (ranked by relevance):
        1. [Score: 87.5] Book A - Highly relevant because...
        2. [Score: 72.3] Book B - Moderately relevant because...
        ...
    """
    import json

    import ast

    try:
        # Parse books from JSON - try multiple strategies
        books = None
        parse_error = None

        # Strategy 1: Direct JSON parsing
        try:
            books = json.loads(books_json)
        except json.JSONDecodeError as e:
            parse_error = e
            logging.warning(f"Direct JSON parse failed: {e}")

            # Strategy 2: Sanitize Python literals to JSON and retry
            try:
                # Replace Python None/True/False with JSON null/true/false
                import re
                sanitized = re.sub(r'\bNone\b', 'null', books_json)
                sanitized = re.sub(r'\bTrue\b', 'true', sanitized)
                sanitized = re.sub(r'\bFalse\b', 'false', sanitized)
                books = json.loads(sanitized)
                logging.info("Successfully parsed after sanitizing Python literals")
            except json.JSONDecodeError as e2:
                logging.warning(f"Sanitized JSON parse failed: {e2}")

                # Strategy 3: Use ast.literal_eval for Python syntax
                try:
                    books = ast.literal_eval(books_json)
                    logging.info("Successfully parsed using ast.literal_eval")
                except (ValueError, SyntaxError) as e3:
                    logging.error(f"All parsing strategies failed: {e3}")
                    raise parse_error  # Re-raise original error for outer handler

        if not books:
            return "ERROR: No books provided for scoring"

        # Get scorer and rank books
        scorer = get_scorer()
        ranked_books = scorer.rank_books(user_prompt, books, return_scores=True)

        # Format output
        output_lines = [f"SCORED BOOKS (ranked by relevance to: '{user_prompt[:80]}...'):"]
        output_lines.append("")

        for i, book in enumerate(ranked_books, 1):
            score = book.get("relevance_score", 0)
            title = book.get("title", "Unknown")
            authors = book.get("authors", ["Unknown"])
            author_str = ", ".join(authors[:2])

            # Categorize relevance
            if score >= 75:
                relevance_note = "HIGHLY relevant"
            elif score >= 60:
                relevance_note = "MODERATELY relevant"
            elif score >= 45:
                relevance_note = "SOMEWHAT relevant"
            else:
                relevance_note = "LOW relevance"

            output_lines.append(
                f"{i}. [Score: {score}] {title} by {author_str} - {relevance_note}"
            )

        output_lines.append("")
        output_lines.append(f"Note: Scores range from 0-100. Scores above 60 indicate good semantic match.")

        return "\n".join(output_lines)

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse books JSON: {e}")
        return f"ERROR: Invalid JSON format - {str(e)}"
    except Exception as e:
        logging.error(f"Error scoring books: {e}")
        return f"ERROR: Failed to score books - {str(e)}"

