"""
Module for interacting with the Open Library API
"""
import requests
from typing import List, Dict, Optional
from config import OPEN_LIBRARY_SEARCH_URL, OPEN_LIBRARY_BASE_URL


class OpenLibraryAPI:
    """Class to interact with Open Library API"""

    @staticmethod
    def search_books(query: str, limit: int = 10) -> List[Dict]:
        """
        Search for books using Open Library API

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of book dictionaries with relevant information
        """
        try:
            params = {
                "q": query,
                "limit": limit,
                "fields": "key,title,author_name,first_publish_year,isbn,subject,publisher,language"
            }

            response = requests.get(OPEN_LIBRARY_SEARCH_URL, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            books = []

            for doc in data.get("docs", []):
                book = {
                    "title": doc.get("title", "Unknown"),
                    "authors": doc.get("author_name", ["Unknown"]),
                    "first_publish_year": doc.get("first_publish_year", "N/A"),
                    "subjects": doc.get("subject", [])[:5],  # Limit subjects
                    "key": doc.get("key", ""),
                    "isbn": doc.get("isbn", [None])[0] if doc.get("isbn") else None,
                }
                books.append(book)

            return books

        except requests.exceptions.RequestException as e:
            print(f"Error fetching books from Open Library: {e}")
            return []

    @staticmethod
    def get_book_details(book_key: str) -> Optional[Dict]:
        """
        Get detailed information about a specific book

        Args:
            book_key: Open Library book key (e.g., "/works/OL45804W")

        Returns:
            Dictionary with book details or None if not found
        """
        try:
            url = f"{OPEN_LIBRARY_BASE_URL}{book_key}.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            return {
                "title": data.get("title", "Unknown"),
                "description": data.get("description", "No description available"),
                "subjects": data.get("subjects", []),
                "key": book_key,
            }

        except requests.exceptions.RequestException as e:
            print(f"Error fetching book details: {e}")
            return None

    @staticmethod
    def search_by_subject(subject: str, limit: int = 10) -> List[Dict]:
        """
        Search for books by subject

        Args:
            subject: Subject to search for
            limit: Maximum number of results

        Returns:
            List of book dictionaries
        """
        return OpenLibraryAPI.search_books(f"subject:{subject}", limit=limit)

    @staticmethod
    def search_by_author(author: str, limit: int = 10) -> List[Dict]:
        """
        Search for books by author

        Args:
            author: Author name
            limit: Maximum number of results

        Returns:
            List of book dictionaries
        """
        return OpenLibraryAPI.search_books(f"author:{author}", limit=limit)

