import logging
import time
import requests
from typing import List, Set, Tuple

from datamodel.search_result import SearchResult
from config import OPEN_LIBRARY_SEARCH_URL, OPEN_LIBRARY_BASE_URL

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


def search_open_library(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    max_results: int = 10,
    author: str | None = None,
    subject: str | None = None,
) -> List[SearchResult]:
    """
    Search Open Library for books.

    Prefers subject-first search. If subject results are sparse, falls back to keyword
    queries. Filters by year client-side and deduplicates results.
    """
    # Apply rate limiting
    _apply_rate_limit()

    def _format_results(docs: List[dict]) -> List[SearchResult]:
        results: List[SearchResult] = []
        seen: Set[Tuple[str, str]] = set()
        for book in docs:
            title = book.get("title", "No title")
            year = book.get("first_publish_year", None)

            # Year filtering
            if year:
                if year_from and year < year_from:
                    continue
                if year_to and year > year_to:
                    continue

            authors = book.get("author_name", [])
            author_names = authors[:3] if authors else ["Unknown"]
            key_for_dedupe = (title.strip().lower(), ",".join([a.strip().lower() for a in author_names]))
            if key_for_dedupe in seen:
                continue
            seen.add(key_for_dedupe)

            isbn_list = book.get("isbn", [])
            isbn = isbn_list[0] if isbn_list else None

            subjects = book.get("subject", [])[:5]

            book_key = book.get("key", "")
            url = f"https://openlibrary.org{book_key}" if book_key else "N/A"

            # Build snippet
            snippet_parts = []
            if subjects:
                snippet_parts.append(f"Subjects: {', '.join(subjects[:3])}")
            publisher = book.get("publisher", [])
            if publisher:
                snippet_parts.append(f"Publisher(s): {', '.join(publisher[:2])}")
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

    # Helper to call API with parameters
    def _call(search_query: str, fetch_limit: int) -> dict | None:
        params = {
            "q": search_query,
            "limit": fetch_limit,
            "fields": "key,title,author_name,first_publish_year,isbn,subject,publisher,language",
        }
        for attempt in range(1, 3):
            logging.info(
                "Calling Open Library API (attempt %d) with query: '%s', year=%s, max_results=%d.",
                attempt,
                search_query,
                (f"{year_from}–{year_to}" if year_from and year_to else (f">= {year_from}" if year_from else (f"<= {year_to}" if year_to else "none"))),
                max_results,
            )
            try:
                resp = requests.get(OPEN_LIBRARY_SEARCH_URL, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                logging.error("Open Library error on attempt %d: %s", attempt, str(e))
                if attempt == 1:
                    time.sleep(1.5)
        return None

    # Subject-first search with synonyms
    subject_variants: List[str] = []
    if subject:
        # Normalize and create variants
        base = subject.strip()
        # Capitalize common genre forms
        subject_variants = [
            base,
            base.title(),
        ]
        # Add helpful synonyms for common genres (minimal set)
        if base.lower() == "fantasy":
            subject_variants.extend(["Fantasy fiction", "High fantasy", "Dark fantasy"])

    # Decide initial fetch size: larger pool to allow client-side filtering
    initial_fetch = max(30, max_results * 5)

    # Try subject-first
    if subject_variants:
        for sv in subject_variants:
            data = _call(search_query=f"subject:{sv}", fetch_limit=initial_fetch)
            docs = (data or {}).get("docs", [])
            results = _format_results(docs)
            if results:
                return results[:max_results]
        # If all subject variants failed, fall back to keyword-only

    # Fallback keyword search
    keyword_queries: List[str] = []
    base_q = (query or "").strip()
    if base_q:
        keyword_queries = [base_q]
        if base_q.lower() == "fantasy":
            keyword_queries.extend(["fantasy novel", "fantasy fiction"])
    else:
        # If no query provided but subject existed and failed, try subject as keyword
        if subject:
            keyword_queries = [subject, subject.title()]

    for kq in keyword_queries:
        data = _call(search_query=kq, fetch_limit=initial_fetch)
        docs = (data or {}).get("docs", [])
        results = _format_results(docs)
        if results:
            return results[:max_results]

    # If nothing found, return empty list
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


def get_book_details(book_url: str) -> dict:
    """
    Fetch detailed information about a book from Open Library using its URL or work ID.

    Args:
        book_url: Full Open Library URL (e.g., 'https://openlibrary.org/works/OL71124W')
                  or just the work ID (e.g., 'OL71124W' or '/works/OL71124W')

    Returns:
        Dictionary with detailed book information including description, subjects, etc.
        Returns error dict if the request fails.
    """
    # Apply rate limiting
    _apply_rate_limit()

    # Extract work ID from URL if full URL provided
    if book_url.startswith('http'):
        # Extract the /works/OLXXXXX part
        if '/works/' in book_url:
            work_id = book_url.split('/works/')[1].split('?')[0]
        elif '/books/' in book_url:
            work_id = book_url.split('/books/')[1].split('?')[0]
        else:
            return {"error": "Invalid Open Library URL format"}
    else:
        # Clean up work ID
        work_id = book_url.replace('/works/', '').replace('/books/', '')

    # Construct the API endpoint
    if work_id.startswith('OL') and work_id[2:].replace('W', '').replace('M', '').isdigit():
        # It's a works or editions ID
        if 'W' in work_id:
            api_url = f"{OPEN_LIBRARY_BASE_URL}/works/{work_id}.json"
        else:
            api_url = f"{OPEN_LIBRARY_BASE_URL}/books/{work_id}.json"
    else:
        return {"error": f"Invalid Open Library ID format: {work_id}"}

    logging.info(f"Fetching book details from: {api_url}")

    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Extract useful information
        result = {
            "title": data.get("title", "N/A"),
            "description": None,
            "subjects": data.get("subjects", []),
            "key": data.get("key", "N/A"),
            "url": f"{OPEN_LIBRARY_BASE_URL}{data.get('key', '')}",
        }

        # Handle description (can be string or dict)
        desc = data.get("description")
        if isinstance(desc, dict):
            result["description"] = desc.get("value", "N/A")
        elif isinstance(desc, str):
            result["description"] = desc
        else:
            result["description"] = "No description available"

        # Add authors if available
        if "authors" in data:
            result["authors"] = [author.get("key", "") for author in data.get("authors", [])]

        # Add publication info if available
        if "publish_date" in data:
            result["publish_date"] = data.get("publish_date")

        if "publishers" in data:
            result["publishers"] = data.get("publishers", [])

        # Add any other useful fields
        if "first_publish_date" in data:
            result["first_publish_date"] = data.get("first_publish_date")

        if "covers" in data:
            result["covers"] = data.get("covers", [])

        logging.info(f"Successfully fetched details for: {result['title']}")
        return result

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching book details from {api_url}: {str(e)}")
        return {"error": f"Failed to fetch book details: {str(e)}"}
