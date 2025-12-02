"""Search result datamodel for papers and books"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SearchResult:
    """
    Unified search result for research papers and books

    Attributes:
        query: The search query that produced this result
        url: URL to the resource
        title: Title of the paper/book
        snippet: Abstract or description
        authors: List of author names
        publication_year: Year of publication (optional)
        citations: Number of citations (optional, for papers)
        isbn: ISBN number (optional, for books)
        subjects: List of subjects/topics (optional, for books)
    """
    query: str
    url: str
    title: str
    snippet: str
    authors: List[str]
    publication_year: Optional[int] = None
    citations: Optional[int] = None
    isbn: Optional[str] = None
    subjects: Optional[List[str]] = None
    volume_id: Optional[str] = None  # Google Books volume ID for reliable detail fetching
    relevance_score: Optional[float] = None  # Semantic relevance score (0-100)

    def __str__(self) -> str:
        """Format the search result as a readable string"""
        result = f"Title: {self.title}\n"
        result += f"Authors: {', '.join(self.authors)}\n"

        if self.publication_year:
            result += f"Year: {self.publication_year}\n"

        if self.citations is not None:
            result += f"Citations: {self.citations}\n"

        if self.isbn:
            result += f"ISBN: {self.isbn}\n"

        if self.subjects and len(self.subjects) > 0:
            result += f"Subjects: {', '.join(self.subjects[:5])}\n"

        if self.relevance_score is not None:
            # Add relevance label (updated thresholds for new scoring formula)
            if self.relevance_score >= 80:
                label = "HIGHLY relevant"
            elif self.relevance_score >= 65:
                label = "MODERATELY relevant"
            elif self.relevance_score >= 50:
                label = "SOMEWHAT relevant"
            else:
                label = "LOW relevance"
            result += f"Relevance Score: {self.relevance_score} ({label})\n"

        result += f"URL: {self.url}\n"
        result += f"Description: {self.snippet}\n"

        return result

