import logging
import time
import requests
from typing import List, Set, Tuple
from utils.recitation_handler import sanitize_book_description
from datamodel.search_result import SearchResult
from config import GOOGLE_BOOKS_API_KEY

# Google Books API Configuration
GOOGLE_BOOKS_BASE_URL = "https://www.googleapis.com/books/v1"
GOOGLE_BOOKS_SEARCH_URL = f"{GOOGLE_BOOKS_BASE_URL}/volumes"

# Simple rate limiting using timestamps
_last_call_time = 0
_min_call_interval = 0.5


def _sanitize_url(url_or_message: str) -> str:
    """Remove API keys from URLs in error messages for security"""
    import re
    # Replace API key with [REDACTED] in URLs
    return re.sub(r'key=[^&\s]+', 'key=[REDACTED]', str(url_or_message))


def _apply_rate_limit():
    """Simple rate limiting function"""
    global _last_call_time
    current_time = time.time()
    time_since_last_call = current_time - _last_call_time

    if time_since_last_call < _min_call_interval:
        sleep_time = _min_call_interval - time_since_last_call
        time.sleep(sleep_time)

    _last_call_time = time.time()



def _extract_volume_id(item: dict = None, url: str = None) -> str | None:
    """
    Extract Google Books volume ID from API item or URL.

    Args:
        item: API response item from Google Books search
        url: Google Books URL string

    Returns:
        Volume ID string or None if extraction fails
    """
    # Try to get ID directly from item first (most reliable)
    if item:
        volume_id = item.get("id")
        if volume_id:
            return volume_id

    # Fall back to URL parsing if no item or ID not found
    if url:
        try:
            # Handle different URL formats:
            # https://books.google.com/books?id=VOLUME_ID&...
            # https://play.google.com/store/books/details?id=VOLUME_ID&...
            # https://www.googleapis.com/books/v1/volumes/VOLUME_ID
            if 'id=' in url:
                volume_id = url.split('id=')[1].split('&')[0]
                return volume_id if volume_id else None
            elif '/volumes/' in url:
                volume_id = url.split('/volumes/')[1].split('?')[0]
                return volume_id if volume_id else None
        except (IndexError, AttributeError):
            pass

    return None


def search_google_books(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    max_results: int = 10,
    author: str | None = None,
    subject: str | None = None,
) -> List[SearchResult]:
    """
    Search Google Books for books.

    Args:
        query: Search query string
        year_from: Minimum publication year (optional)
        year_to: Maximum publication year (optional)
        max_results: Maximum number of results to return (default: 10, max: 40)
        author: Author name filter (optional)
        subject: Subject/genre filter (optional)

    Returns:
        List of SearchResult objects
    """
    # Apply rate limiting
    _apply_rate_limit()

    if not GOOGLE_BOOKS_API_KEY:
        logging.error("GOOGLE_BOOKS_API_KEY is not set in environment variables")
        return []

    def _format_results(items: List[dict]) -> List[SearchResult]:
        """Format Google Books API results into SearchResult objects"""
        results: List[SearchResult] = []
        seen: Set[Tuple[str, str]] = set()

        for item in items:
            volume_info = item.get("volumeInfo", {})

            title = volume_info.get("title", "No title")

            # Extract publication year
            published_date = volume_info.get("publishedDate", "")
            year = None
            if published_date:
                try:
                    # publishedDate can be YYYY, YYYY-MM, or YYYY-MM-DD
                    year = int(published_date.split("-")[0])
                except (ValueError, IndexError):
                    year = None

            # Year filtering
            if year:
                if year_from and year < year_from:
                    continue
                if year_to and year > year_to:
                    continue

            # Extract authors
            authors = volume_info.get("authors", [])
            author_names = authors[:3] if authors else ["Unknown"]

            # Deduplication
            key_for_dedupe = (title.strip().lower(), ",".join([a.strip().lower() for a in author_names]))
            if key_for_dedupe in seen:
                continue
            seen.add(key_for_dedupe)

            # Extract ISBN
            isbn = None
            industry_identifiers = volume_info.get("industryIdentifiers", [])
            for identifier in industry_identifiers:
                if identifier.get("type") in ["ISBN_13", "ISBN_10"]:
                    isbn = identifier.get("identifier")
                    break

            # Extract subjects/categories
            subjects = volume_info.get("categories", [])

            # Get URL
            url = volume_info.get("infoLink", volume_info.get("previewLink", "N/A"))

            # Extract volume ID for reliable detail fetching later
            volume_id = _extract_volume_id(item=item, url=url)

            # Skip items without valid volume ID (they won't work for detail fetching)
            if not volume_id:
                logging.warning(f"Skipping book '{title}' - no valid volume ID found")
                continue

            # Build snippet
            snippet = volume_info.get("description", "")
            if not snippet:
                # Fallback to search snippet if description not available
                search_info = item.get("searchInfo", {})
                snippet = search_info.get("textSnippet", "No description available")

            # Sanitize snippet to avoid RECITATION errors when agent quotes it
            snippet = sanitize_book_description(snippet, max_length=300)

            # Truncate snippet if too long
            if len(snippet) > 500:
                snippet = snippet[:497] + "..."

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
                    volume_id=volume_id,
                )
            )
        return results

    # Build query string for Google Books API
    query_parts = []

    # Add base query
    if query and query.strip():
        query_parts.append(query.strip())

    # Add subject filter
    if subject:
        query_parts.append(f"subject:{subject}")

    # Add author filter
    if author:
        query_parts.append(f"inauthor:{author}")

    # Combine query parts
    search_query = " ".join(query_parts) if query_parts else "fiction"

    # Build API parameters
    params = {
        "q": search_query,
        "maxResults": min(max_results, 40),  # Google Books API max is 40
        "key": GOOGLE_BOOKS_API_KEY,
        "printType": "books",
        "langRestrict": "en",
    }

    # Helper to call API with retry logic
    def _call(attempt: int = 1) -> dict | None:
        logging.info(
            "Calling Google Books API (attempt %d) with query: '%s', year=%s, max_results=%d.",
            attempt,
            search_query,
            (f"{year_from}–{year_to}" if year_from and year_to else (f">= {year_from}" if year_from else (f"<= {year_to}" if year_to else "none"))),
            max_results,
        )
        try:
            resp = requests.get(GOOGLE_BOOKS_SEARCH_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logging.error("Google Books API error on attempt %d: %s", attempt, _sanitize_url(str(e)))
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

    # Format and return results
    items = data.get("items", [])
    results = _format_results(items)
    return results[:max_results]


def get_book_details_google(volume_id: str) -> dict:
    """
    Fetch detailed information about a book from Google Books using its volume ID.

    Args:
        volume_id: Google Books volume ID (e.g., 'zyTCAlFPjgYC')
                   Can also be a full URL from Google Books

    Returns:
        Dictionary with detailed book information including description, subjects, etc.
        Returns error dict with 'error' and 'error_type' if the request fails.
    """

    # Apply rate limiting
    _apply_rate_limit()

    if not GOOGLE_BOOKS_API_KEY:
        logging.error("GOOGLE_BOOKS_API_KEY is not set in environment variables")
        return {
            "error": "GOOGLE_BOOKS_API_KEY is not configured",
            "error_type": "config_error"
        }

    # Extract volume ID from URL if full URL provided
    original_input = volume_id
    if volume_id.startswith('http'):
        extracted_id = _extract_volume_id(url=volume_id)
        if not extracted_id:
            return {
                "error": f"Invalid Google Books URL format: {volume_id}",
                "error_type": "invalid_url"
            }
        volume_id = extracted_id

    # Validate volume ID format (basic check)
    if not volume_id or len(volume_id) < 8:
        return {
            "error": f"Invalid volume ID format: {volume_id}",
            "error_type": "invalid_id"
        }

    # Construct the API endpoint
    api_url = f"{GOOGLE_BOOKS_SEARCH_URL}/{volume_id}"
    params = {"key": GOOGLE_BOOKS_API_KEY}

    logging.info(f"Fetching book details from Google Books API: {volume_id}")

    # Retry logic with exponential backoff
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        status_code = None  # Initialize to None for each attempt
        try:
            resp = requests.get(api_url, params=params, timeout=15)
            status_code = resp.status_code
            resp.raise_for_status()
            data = resp.json()

            volume_info = data.get("volumeInfo", {})

            # Extract useful information
            result = {
                "title": volume_info.get("title", "N/A"),
                "subtitle": volume_info.get("subtitle"),
                "description": volume_info.get("description", "No description available"),
                "authors": volume_info.get("authors", []),
                "publisher": volume_info.get("publisher"),
                "publishedDate": volume_info.get("publishedDate"),
                "categories": volume_info.get("categories", []),
                "pageCount": volume_info.get("pageCount"),
                "language": volume_info.get("language"),
                "previewLink": volume_info.get("previewLink"),
                "infoLink": volume_info.get("infoLink"),
                "volumeId": data.get("id"),
            }

            # Extract ISBN
            industry_identifiers = volume_info.get("industryIdentifiers", [])
            for identifier in industry_identifiers:
                id_type = identifier.get("type")
                if id_type == "ISBN_13":
                    result["isbn_13"] = identifier.get("identifier")
                elif id_type == "ISBN_10":
                    result["isbn_10"] = identifier.get("identifier")

            # Extract image links
            image_links = volume_info.get("imageLinks", {})
            if image_links:
                result["thumbnail"] = image_links.get("thumbnail")
                result["smallThumbnail"] = image_links.get("smallThumbnail")

            # Add rating information if available
            if "averageRating" in volume_info:
                result["averageRating"] = volume_info.get("averageRating")
                result["ratingsCount"] = volume_info.get("ratingsCount")

            # Sanitize description to avoid RECITATION errors in Google Gemini
            # RECITATION errors occur when Gemini detects copyrighted content
            try:
                from utils.recitation_handler import sanitize_book_details_for_gemini
                result = sanitize_book_details_for_gemini(result)
                logging.debug(f"Sanitized book description for Gemini compatibility")
            except ImportError:
                logging.warning("Could not import recitation_handler, skipping sanitization")
            except Exception as e:
                logging.warning(f"Failed to sanitize book details: {e}")

            # Log successful fetch
            logging.info(f"Successfully fetched details for volume {volume_id} -> '{result['title']}' by {result['authors']}")
            return result

        except requests.exceptions.HTTPError as e:
            # Try to extract status code from response, fallback to status_code from before raise_for_status
            if e.response is not None:
                status_code = e.response.status_code
            # If status_code is still None, this is unexpected

            # 404 means the book doesn't exist - don't retry
            if status_code == 404:
                logging.warning(f"Book not found (404) for volume ID: {volume_id}")
                return {
                    "error": f"Book not found for volume ID: {volume_id}",
                    "error_type": "not_found",
                    "volume_id": volume_id,
                    "status_code": 404
                }

            # 503 Service Unavailable - retry with longer backoff
            if status_code == 503:
                if attempt < max_retries:
                    # Longer exponential backoff for 503: 3, 6, 9 seconds
                    backoff_time = 3 * attempt
                    logging.warning(
                        f"Service unavailable (503) for {volume_id}, "
                        f"retrying in {backoff_time}s (attempt {attempt}/{max_retries})"
                    )
                    time.sleep(backoff_time)
                    continue
                else:
                    logging.error(f"Service unavailable (503) for {volume_id} after {max_retries} attempts")
                    return {
                        "error": f"Google Books API service unavailable after {max_retries} attempts",
                        "error_type": "service_unavailable",
                        "volume_id": volume_id,
                        "status_code": 503
                    }

            # Other HTTP errors - sanitize the error message
            sanitized_error = _sanitize_url(str(e))
            logging.error(
                f"HTTP error {status_code} fetching details for {volume_id}: {sanitized_error}"
            )
            return {
                "error": f"HTTP {status_code} error: {sanitized_error}",
                "error_type": "http_error",
                "volume_id": volume_id,
                "status_code": status_code
            }

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                backoff_time = 2 ** attempt
                logging.warning(
                    f"Timeout fetching {volume_id}, "
                    f"retrying in {backoff_time}s (attempt {attempt}/{max_retries})"
                )
                time.sleep(backoff_time)
                continue
            else:
                logging.error(f"Timeout fetching details for {volume_id} after {max_retries} attempts")
                return {
                    "error": f"Request timeout after {max_retries} attempts",
                    "error_type": "timeout",
                    "volume_id": volume_id
                }

        except requests.exceptions.RequestException as e:
            sanitized_error = _sanitize_url(str(e))
            logging.error(f"Request error fetching details for {volume_id}: {sanitized_error}")
            return {
                "error": f"Request failed: {sanitized_error}",
                "error_type": "request_error",
                "volume_id": volume_id
            }

    # If we exhausted all retries
    return {
        "error": f"Failed to fetch book details after {max_retries} attempts",
        "error_type": "max_retries_exceeded",
        "volume_id": volume_id
    }


