import logging
import time
import requests
from typing import List, Set, Tuple

from datamodel.search_result import SearchResult
from config import BIG_BOOK_API_KEY

# Big Book API Configuration
BIG_BOOK_BASE_URL = "https://api.bigbookapi.com"

# Simple rate limiting using timestamps
_last_call_time = 0
_min_call_interval = 1.0  # Minimum 1 second between calls


def _apply_rate_limit():
    """Simple rate limiting function"""
    global _last_call_time
    current_time = time.time()
    time_since_last_call = current_time - _last_call_time

    if time_since_last_call < _min_call_interval:
        sleep_time = _min_call_interval - time_since_last_call
        time.sleep(sleep_time)

    _last_call_time = time.time()


def search_big_book(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    max_results: int = 10,
    author: str | None = None,
    subject: str | None = None,
) -> List[SearchResult]:
    """
    Search Big Book API for books.

    Args:
        query: Search query string
        year_from: Minimum publication year (optional)
        year_to: Maximum publication year (optional)
        max_results: Maximum number of results to return (default: 10)
        author: Author name filter (optional)
        subject: Subject/genre filter (optional)

    Returns:
        List of SearchResult objects
    """
    # Apply rate limiting
    _apply_rate_limit()

    if not BIG_BOOK_API_KEY:
        logging.error("BIG_BOOK_API_KEY is not set in environment variables")
        return []

    def _format_results(books: List[dict]) -> List[SearchResult]:
        """Format Big Book API results into SearchResult objects"""
        results: List[SearchResult] = []
        seen: Set[Tuple[str, str]] = set()

        for book in books:
            title = book.get("title", "No title")

            # Extract publication year
            publish_date = book.get("publish_date", "") or book.get("year", "")
            year = None
            if publish_date:
                try:
                    # Try to extract year from various date formats
                    if isinstance(publish_date, int):
                        year = publish_date
                    elif isinstance(publish_date, str):
                        # Try to find a 4-digit year
                        import re
                        year_match = re.search(r'\b(19|20)\d{2}\b', publish_date)
                        if year_match:
                            year = int(year_match.group())
                except (ValueError, AttributeError):
                    year = None

            # Year filtering
            if year:
                if year_from and year < year_from:
                    continue
                if year_to and year > year_to:
                    continue

            # Extract authors
            authors = book.get("authors", [])
            if isinstance(authors, str):
                authors = [authors]
            author_names = authors[:3] if authors else ["Unknown"]

            # Deduplication
            key_for_dedupe = (title.strip().lower(), ",".join([a.strip().lower() for a in author_names]))
            if key_for_dedupe in seen:
                continue
            seen.add(key_for_dedupe)

            # Extract ISBN
            isbn = book.get("isbn13") or book.get("isbn10") or book.get("isbn")

            # Extract subjects/categories
            subjects = book.get("categories", []) or book.get("subjects", [])
            if isinstance(subjects, str):
                subjects = [subjects]

            # Get URL
            url = book.get("info_url") or book.get("preview_url") or "N/A"

            # Build snippet from description
            description = book.get("description", "") or book.get("summary", "")
            if description:
                # Truncate if too long
                snippet = description[:500] + "..." if len(description) > 500 else description
            else:
                # Fallback snippet from other fields
                snippet_parts = []
                if subjects:
                    snippet_parts.append(f"Categories: {', '.join(subjects[:3])}")
                publisher = book.get("publisher")
                if publisher:
                    snippet_parts.append(f"Publisher: {publisher}")
                snippet = ". ".join(snippet_parts) if snippet_parts else "No description available"

            results.append(
                SearchResult(
                    query=query,
                    url=url,
                    title=title,
                    snippet=snippet,
                    authors=author_names,
                    publication_year=year,
                    isbn=isbn,
                    subjects=subjects,
                )
            )
        return results

    # Build query parameters for Big Book API
    # The API endpoint is typically: /search?q=query&api-key=key
    search_url = f"{BIG_BOOK_BASE_URL}/search"

    params = {
        "api-key": BIG_BOOK_API_KEY,
        "number": max_results,
    }

    # Build search query
    query_parts = []
    if query and query.strip():
        query_parts.append(query.strip())

    if author:
        query_parts.append(author)

    if subject:
        query_parts.append(subject)

    # Combine query parts
    search_query = " ".join(query_parts) if query_parts else "fiction"
    params["q"] = search_query

    # Helper to call API with retry logic
    def _call(attempt: int = 1) -> dict | None:
        logging.info(
            "Calling Big Book API (attempt %d) with query: '%s', year=%s, max_results=%d.",
            attempt,
            search_query,
            (f"{year_from}–{year_to}" if year_from and year_to else (f">= {year_from}" if year_from else (f"<= {year_to}" if year_to else "none"))),
            max_results,
        )
        try:
            resp = requests.get(search_url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logging.error("Big Book API error on attempt %d: %s", attempt, str(e))
            if attempt < 3:
                sleep_time = 1.5 * attempt  # Exponential backoff
                time.sleep(sleep_time)
                return _call(attempt + 1)
        return None

    # Make API call
    data = _call()
    if not data:
        logging.info(
            "No results found for query: '%s' with filters year_from=%s, year_to=%s, author=%s, subject=%s, max_results=%d.",
            query,
            year_from,
            year_to,
            author,
            subject,
            max_results,
        )
        return []

    # Extract books from response
    # The response structure may vary, common patterns:
    books = data.get("books", []) or data.get("results", []) or data.get("items", [])

    # If data is directly a list
    if isinstance(data, list):
        books = data

    # Format and return results
    results = _format_results(books)
    return results[:max_results]


def get_book_details_big_book(book_id: str) -> dict:
    """
    Fetch detailed information about a book from Big Book API using its book ID.

    Args:
        book_id: Big Book API book ID or ISBN
                 Can also be a full URL

    Returns:
        Dictionary with detailed book information including description, subjects, etc.
        Returns error dict if the request fails.
    """
    # Apply rate limiting
    _apply_rate_limit()

    if not BIG_BOOK_API_KEY:
        logging.error("BIG_BOOK_API_KEY is not set in environment variables")
        return {"error": "BIG_BOOK_API_KEY is not configured"}

    # Extract book ID from URL if full URL provided
    if book_id.startswith('http'):
        # Try to extract ID from various URL patterns
        parts = book_id.rstrip('/').split('/')
        book_id = parts[-1] if parts else book_id

    # Construct the API endpoint
    # Big Book API typically uses: /book_id or /isbn/book_id
    api_url = f"{BIG_BOOK_BASE_URL}/{book_id}"
    params = {"api-key": BIG_BOOK_API_KEY}

    logging.info(f"Fetching book details from Big Book API: {book_id}")

    try:
        resp = requests.get(api_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Extract useful information
        result = {
            "id": data.get("id") or book_id,
            "title": data.get("title", "N/A"),
            "subtitle": data.get("subtitle"),
            "description": data.get("description") or data.get("summary", "No description available"),
            "authors": data.get("authors", []),
            "publisher": data.get("publisher"),
            "publish_date": data.get("publish_date") or data.get("year"),
            "categories": data.get("categories", []) or data.get("subjects", []),
            "page_count": data.get("page_count") or data.get("pages"),
            "language": data.get("language"),
            "isbn13": data.get("isbn13"),
            "isbn10": data.get("isbn10"),
            "info_url": data.get("info_url"),
            "preview_url": data.get("preview_url"),
        }

        # Extract image/cover information if available
        if "image" in data:
            result["image"] = data.get("image")
        if "cover_url" in data:
            result["cover_url"] = data.get("cover_url")
        if "thumbnail" in data:
            result["thumbnail"] = data.get("thumbnail")

        # Add rating information if available
        if "rating" in data:
            result["rating"] = data.get("rating")
        if "ratings_count" in data:
            result["ratings_count"] = data.get("ratings_count")

        logging.info(f"Successfully fetched details for: {result['title']}")
        return result

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching book details from Big Book API: {str(e)}")
        return {"error": f"Failed to fetch book details: {str(e)}"}

