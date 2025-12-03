"""Utility to sanitize book descriptions to avoid RECITATION errors in Google Gemini."""
import re


def sanitize_book_description(description: str, max_length: int = 300) -> str:
    """
    Sanitize book description to avoid RECITATION errors from Google Gemini.

    RECITATION errors occur when Gemini detects copyrighted content in the input.
    This happens frequently with full book descriptions from Google Books.

    Strategy:
    1. Truncate long descriptions
    2. Remove HTML tags
    3. Paraphrase or summarize instead of quoting verbatim

    Args:
        description: Original book description
        max_length: Maximum length for sanitized description

    Returns:
        Sanitized description text
    """
    if not description or description == "No description available":
        return "No description available"

    # Remove HTML tags
    clean_desc = re.sub(r'<[^>]+>', '', description)

    # Remove extra whitespace
    clean_desc = ' '.join(clean_desc.split())

    # Truncate if too long
    if len(clean_desc) > max_length:
        clean_desc = clean_desc[:max_length].rsplit(' ', 1)[0] + "..."

    return clean_desc


def sanitize_book_details_for_gemini(book_details: dict) -> dict:
    """
    Sanitize book details dictionary to avoid RECITATION errors.

    Args:
        book_details: Dict with keys like title, description, authors, etc.

    Returns:
        Sanitized book details dict
    """
    sanitized = book_details.copy()

    # Sanitize description
    if 'description' in sanitized:
        sanitized['description'] = sanitize_book_description(sanitized['description'])

    # Keep other fields as-is (they're typically safe)
    return sanitized


def create_brief_summary(book_details: dict) -> str:
    """
    Create a brief, non-copyrighted summary of a book from its details.

    Use this when full description causes RECITATION errors.

    Args:
        book_details: Book details dict

    Returns:
        Brief summary string
    """
    title = book_details.get('title', 'Unknown')
    authors = book_details.get('authors', ['Unknown'])
    year = book_details.get('publishedDate', 'Unknown')
    publisher = book_details.get('publisher', 'Unknown')
    categories = book_details.get('categories', [])
    page_count = book_details.get('pageCount', 'Unknown')

    # Format authors
    if isinstance(authors, list):
        authors_str = ', '.join(authors)
    else:
        authors_str = str(authors)

    # Format categories
    if isinstance(categories, list) and categories:
        categories_str = ', '.join(categories)
    else:
        categories_str = 'General'

    # Create brief summary without quoting copyrighted text
    summary = (
        f"This {page_count}-page book titled '{title}' by {authors_str} "
        f"was published by {publisher} in {year}. "
        f"It falls under the category of {categories_str}."
    )

    return summary


def format_book_without_description(book_details: dict) -> str:
    """
    Format book details without including full description to avoid RECITATION.

    Args:
        book_details: Book details dict

    Returns:
        Formatted string with book info
    """
    title = book_details.get('title', 'Unknown')
    authors = book_details.get('authors', ['Unknown'])
    year = book_details.get('publishedDate', 'Unknown')[:4]  # Just year
    publisher = book_details.get('publisher', 'Unknown')
    categories = book_details.get('categories', [])
    page_count = book_details.get('pageCount', 'Unknown')
    url = book_details.get('infoLink', book_details.get('previewLink', 'No URL'))

    # Format authors
    if isinstance(authors, list):
        authors_str = ', '.join(authors)
    else:
        authors_str = str(authors)

    # Format categories
    if isinstance(categories, list) and categories:
        categories_str = ', '.join(categories)
    else:
        categories_str = 'Not specified'

    # Create formatted output
    output = f"""Title: {title}
Authors: {authors_str}
Year: {year}
Publisher: {publisher}
Categories: {categories_str}
Page Count: {page_count}
URL: {url}
Summary: {create_brief_summary(book_details)}"""

    return output


if __name__ == "__main__":
    # Test with sample book data
    test_book = {
        'title': 'Test Book',
        'authors': ['John Doe', 'Jane Smith'],
        'publishedDate': '2023-01-15',
        'publisher': 'Test Publisher',
        'categories': ['Fiction', 'Science Fiction'],
        'pageCount': 350,
        'description': '<p>This is a <b>very long description</b> with lots of HTML tags that might trigger RECITATION errors because it contains copyrighted text from the publisher...</p>' * 10,
        'infoLink': 'https://books.google.com/example'
    }

    print("Original description length:", len(test_book['description']))
    print("\nSanitized description:")
    print(sanitize_book_description(test_book['description']))
    print("\nBrief summary:")
    print(create_brief_summary(test_book))
    print("\nFormatted without description:")
    print(format_book_without_description(test_book))

