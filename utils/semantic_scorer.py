"""
Semantic scoring module for book recommendations using sentence embeddings.
Uses sentence-transformers to calculate semantic similarity between user prompts and book descriptions.
"""
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util


class SemanticScorer:
    """
    Calculates semantic relevance scores for books based on user prompts.
    Uses all-MiniLM-L6-v2 model for fast, efficient sentence embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic scorer with a sentence transformer model.

        Args:
            model_name: HuggingFace model name. Default is all-MiniLM-L6-v2 (fast, 80MB).
                       Alternative: all-mpnet-base-v2 (more accurate, 420MB)
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        logging.info(f"SemanticScorer initialized with model: {model_name}")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model only when needed."""
        if self._model is None:
            logging.info(f"Loading sentence transformer model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            logging.info("Model loaded successfully")
        return self._model

    def _prepare_book_text(self, book: Dict[str, Any]) -> str:
        """
        Prepare book text for embedding by combining relevant fields.

        Args:
            book: Dictionary containing book information

        Returns:
            Combined text string for embedding
        """
        parts = []

        # Add title (most important)
        if book.get("title"):
            parts.append(f"Title: {book['title']}")

        # Add description/snippet (most informative)
        description = book.get("description") or book.get("snippet", "")
        if description and description != "No description available":
            # Truncate very long descriptions to avoid embedding limits
            if len(description) > 1000:
                description = description[:1000] + "..."
            parts.append(description)

        # Add subjects/genres (important for filtering)
        subjects = book.get("subjects", [])
        if subjects:
            subjects_str = ", ".join(subjects[:5])  # Limit to first 5 subjects
            parts.append(f"Genres: {subjects_str}")

        # Add authors (can be relevant for style/theme)
        authors = book.get("authors", [])
        if authors:
            authors_str = ", ".join(authors[:3])  # Limit to first 3 authors
            parts.append(f"Authors: {authors_str}")

        return " | ".join(parts) if parts else "No information available"

    def score_book(self, user_prompt: str, book: Dict[str, Any]) -> float:
        """
        Calculate semantic similarity score between user prompt and book.

        Args:
            user_prompt: User's search query/requirements
            book: Dictionary containing book information

        Returns:
            Relevance score from 0-100 (higher is more relevant)
        """
        # Prepare texts for embedding
        book_text = self._prepare_book_text(book)

        # Generate embeddings
        prompt_embedding = self.model.encode(user_prompt, convert_to_tensor=True)
        book_embedding = self.model.encode(book_text, convert_to_tensor=True)

        # Calculate cosine similarity (returns 0-1 for text embeddings)
        similarity = util.cos_sim(prompt_embedding, book_embedding).item()

        # Convert to 0-100 scale with better distribution
        # Apply power function to spread scores and emphasize good matches
        # similarity^0.7 gives: 0.5→59, 0.6→68, 0.7→77, 0.8→85, 0.9→93
        score = (similarity ** 0.7) * 100

        return round(score, 2)

    def rank_books(
        self,
        user_prompt: str,
        books: List[Dict[str, Any]],
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rank books by semantic relevance to user prompt.

        Args:
            user_prompt: User's search query/requirements
            books: List of book dictionaries
            return_scores: If True, add 'relevance_score' field to each book

        Returns:
            List of books sorted by relevance (highest first), optionally with scores
        """
        if not books:
            return []

        logging.info(f"Ranking {len(books)} books for prompt: {user_prompt[:100]}...")

        # Calculate scores for all books
        scored_books = []
        for book in books:
            score = self.score_book(user_prompt, book)

            if return_scores:
                # Add score to book dict (create copy to avoid modifying original)
                book_with_score = book.copy()
                book_with_score["relevance_score"] = score
                scored_books.append(book_with_score)
            else:
                scored_books.append((score, book))

        # Sort by score (highest first)
        if return_scores:
            scored_books.sort(key=lambda x: x["relevance_score"], reverse=True)
            logging.info(f"Top 3 scores: {[b['relevance_score'] for b in scored_books[:3]]}")
        else:
            scored_books.sort(key=lambda x: x[0], reverse=True)
            scored_books = [book for _, book in scored_books]

        return scored_books

    def get_top_books(
        self,
        user_prompt: str,
        books: List[Dict[str, Any]],
        top_n: int = 3,
        min_score: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Get top N most relevant books, optionally filtered by minimum score.

        Args:
            user_prompt: User's search query/requirements
            books: List of book dictionaries
            top_n: Number of top books to return
            min_score: Minimum relevance score (0-100) to include a book

        Returns:
            List of top N books with relevance scores
        """
        ranked_books = self.rank_books(user_prompt, books, return_scores=True)

        # Filter by minimum score
        filtered_books = [b for b in ranked_books if b["relevance_score"] >= min_score]

        # Return top N
        return filtered_books[:top_n]


# Global instance for reuse (avoid reloading model)
_scorer_instance: Optional[SemanticScorer] = None


def get_scorer() -> SemanticScorer:
    """
    Get or create global SemanticScorer instance.
    Uses singleton pattern to avoid reloading model multiple times.
    """
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = SemanticScorer()
    return _scorer_instance


def calculate_relevance_scores(
    user_prompt: str,
    books: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convenience function to calculate and add relevance scores to books.

    Args:
        user_prompt: User's search query/requirements
        books: List of book dictionaries

    Returns:
        Books with added 'relevance_score' field, sorted by relevance
    """
    scorer = get_scorer()
    return scorer.rank_books(user_prompt, books, return_scores=True)

