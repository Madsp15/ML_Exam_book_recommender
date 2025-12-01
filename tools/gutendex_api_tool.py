import logging
import time
import requests
from typing import List, Set, Tuple

from datamodel.search_result import SearchResult

# Gutendex API Configuration
GUTENDEX_BASE_URL = "https://gutendex.com"
GUTENDEX_SEARCH_URL = f"{GUTENDEX_BASE_URL}/books"

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


def search_gutendex(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    max_results: int = 10,
    author: str | None = None,
    subject: str | None = None,
) -> List[SearchResult]:
    """
    Search Gutendex (Project Gutenberg) for free public domain books.

    Args:
        query: Search query string (searches in title)
        year_from: Minimum publication year (optional)
        year_to: Maximum publication year (optional)
        max_results: Maximum number of results to return (default: 10)
        author: Author name filter (optional)
        subject: Subject/topic filter (optional)

    Returns:
        List of SearchResult objects
    """
    # Apply rate limiting
    _apply_rate_limit()

    def _format_results(books: List[dict]) -> List[SearchResult]:
        """Format Gutendex API results into SearchResult objects"""
        results: List[SearchResult] = []
        seen: Set[Tuple[str, str]] = set()

        for book in books:
            title = book.get("title", "No title")

            # Extract publication year (Gutendex doesn't always have this)
            # Try to extract from copyright field or use None
            year = None
            copyright_status = book.get("copyright", False)
            # Most Gutenberg books are pre-1928, but we don't have exact dates
            # So we'll skip year filtering for Gutendex unless provided in metadata

            # Year filtering (if we have the year)
            if year:
                if year_from and year < year_from:
                    continue
                if year_to and year > year_to:
                    continue

            # Extract authors
            authors_data = book.get("authors", [])
            author_names = []
            for author in authors_data[:3]:
                name = author.get("name", "Unknown")
                author_names.append(name)
            if not author_names:
                author_names = ["Unknown"]

            # Deduplication
            key_for_dedupe = (title.strip().lower(), ",".join([a.strip().lower() for a in author_names]))
            if key_for_dedupe in seen:
                continue
            seen.add(key_for_dedupe)

            # Extract subjects/bookshelves
            subjects = book.get("subjects", [])[:5]
            if not subjects:
                subjects = book.get("bookshelves", [])[:5]

            # Get URL - use the Gutendex API URL or HTML page
            book_id = book.get("id")
            url = f"https://www.gutenberg.org/ebooks/{book_id}" if book_id else "N/A"

            # Build snippet from subjects and languages
            snippet_parts = []
            if subjects:
                snippet_parts.append(f"Subjects: {', '.join(subjects[:3])}")

            languages = book.get("languages", [])
            if languages:
                snippet_parts.append(f"Languages: {', '.join(languages)}")

            download_count = book.get("download_count", 0)
            if download_count:
                snippet_parts.append(f"Downloads: {download_count}")

            snippet = ". ".join(snippet_parts) if snippet_parts else "Free public domain book from Project Gutenberg"

            # Note: Gutenberg books don't have ISBNs (public domain), so we'll leave it as None
            results.append(
                SearchResult(
                    query=query,
                    url=url,
                    title=title,
                    snippet=snippet,
                    authors=author_names,
                    publication_year=year,
                    isbn=None,  # Public domain books don't have ISBNs
                    subjects=subjects,
                )
            )
        return results

    # Build API parameters for Gutendex
    params = {}

    # Add search query (title search)
    if query and query.strip():
        params["search"] = query.strip()

    # Add topic filter
    if subject:
        params["topic"] = subject

    # Add author filter (author name search)
    if author:
        # Gutendex uses author name in the author filter
        # We'll need to search by author name
        # The API doesn't have direct author search, so we'll filter results
        pass  # Will filter in results

    # Helper to call API with retry logic
    def _call(attempt: int = 1, page: int = 1) -> dict | None:
        current_params = params.copy()
        current_params["page"] = page

        logging.info(
            "Calling Gutendex API (attempt %d, page %d) with query: '%s', author=%s, subject=%s, max_results=%d.",
            attempt,
            page,
            query or "all",
            author or "none",
            subject or "none",
            max_results,
        )
        try:
            resp = requests.get(GUTENDEX_SEARCH_URL, params=current_params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logging.error("Gutendex API error on attempt %d: %s", attempt, str(e))
            if attempt < 3:
                sleep_time = 1.5 * attempt  # Exponential backoff
                time.sleep(sleep_time)
                return _call(attempt + 1, page)
        return None

    # Make API call(s) - may need multiple pages
    all_results: List[SearchResult] = []
    page = 1

    while len(all_results) < max_results:
        data = _call(page=page)
        if not data:
            break

        books = data.get("results", [])
        if not books:
            break

        # Filter by author if specified (client-side filtering)
        if author:
            filtered_books = []
            author_lower = author.lower()
            for book in books:
                book_authors = book.get("authors", [])
                for book_author in book_authors:
                    author_name = book_author.get("name", "").lower()
                    if author_lower in author_name or author_name in author_lower:
                        filtered_books.append(book)
                        break
            books = filtered_books

        # Format results
        results = _format_results(books)
        all_results.extend(results)

        # Check if we have more pages
        if not data.get("next") or len(all_results) >= max_results:
            break

        page += 1
        _apply_rate_limit()  # Rate limit between pages

    if not all_results:
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

    return all_results[:max_results]


def get_book_details_gutendex(book_id: str | int) -> dict:
    """
    Fetch detailed information about a book from Gutendex using its book ID.

    Args:
        book_id: Gutenberg book ID (e.g., 1342 for "Pride and Prejudice")
                 Can also be a full URL from Project Gutenberg

    Returns:
        Dictionary with detailed book information including subjects, formats, etc.
        Returns error dict if the request fails.
    """
    # Apply rate limiting
    _apply_rate_limit()

    # Extract book ID from URL if full URL provided
    if isinstance(book_id, str) and book_id.startswith('http'):
        # Extract book ID from URL patterns like:
        # https://www.gutenberg.org/ebooks/1342
        # https://gutendex.com/books/1342
        if '/ebooks/' in book_id:
            book_id = book_id.split('/ebooks/')[1].split('/')[0].split('?')[0]
        elif '/books/' in book_id:
            book_id = book_id.split('/books/')[1].split('/')[0].split('?')[0]
        else:
            return {"error": "Invalid Gutenberg URL format"}

    # Construct the API endpoint
    api_url = f"{GUTENDEX_SEARCH_URL}/{book_id}"

    logging.info(f"Fetching book details from Gutendex API: {book_id}")

    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Extract useful information
        result = {
            "id": data.get("id"),
            "title": data.get("title", "N/A"),
            "authors": [],
            "translators": [],
            "subjects": data.get("subjects", []),
            "bookshelves": data.get("bookshelves", []),
            "languages": data.get("languages", []),
            "copyright": data.get("copyright"),
            "media_type": data.get("media_type"),
            "download_count": data.get("download_count"),
            "formats": data.get("formats", {}),
            "gutenberg_url": f"https://www.gutenberg.org/ebooks/{data.get('id')}",
        }

        # Extract authors with birth/death years
        authors_data = data.get("authors", [])
        for author in authors_data:
            author_info = {
                "name": author.get("name"),
                "birth_year": author.get("birth_year"),
                "death_year": author.get("death_year"),
            }
            result["authors"].append(author_info)

        # Extract translators
        translators_data = data.get("translators", [])
        for translator in translators_data:
            translator_info = {
                "name": translator.get("name"),
                "birth_year": translator.get("birth_year"),
                "death_year": translator.get("death_year"),
            }
            result["translators"].append(translator_info)

        # Build a description from available metadata
        description_parts = []
        if result["subjects"]:
            description_parts.append(f"Subjects: {', '.join(result['subjects'][:5])}")
        if result["bookshelves"]:
            description_parts.append(f"Bookshelves: {', '.join(result['bookshelves'][:3])}")
        if result["download_count"]:
            description_parts.append(f"Download count: {result['download_count']}")

        result["description"] = ". ".join(description_parts) if description_parts else "Free public domain book from Project Gutenberg"

        # List available formats
        formats = result["formats"]
        result["available_formats"] = list(formats.keys()) if formats else []

        logging.info(f"Successfully fetched details for: {result['title']}")
        return result

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching book details from Gutendex API: {str(e)}")
        return {"error": f"Failed to fetch book details: {str(e)}"}

