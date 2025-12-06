# Book Recommender Agent System

An intelligent multi-agent system for book discovery and recommendations, powered by AutoGen framework and LLMs. The system uses a two-stage workflow with semantic scoring to find the most relevant books based on user requirements.

## Features

- **Multi-Agent Architecture**: Collaborative agents (Librarian, Internal Critic, User Proxy) work together
- **Two-Stage Search**: Initial screening followed by detailed evaluation
- **Semantic Scoring**: Uses sentence-transformers for intelligent relevance ranking
- **Multiple LLM Support**: Google Gemini, Cerebras, and Mistral AI
- **Web Interface**: Interactive Streamlit UI for real-time results
- **CLI Mode**: Batch processing for testing and evaluation
- **Error Recovery**: Robust handling of API failures and content policy issues
- **Smart Filtering**: Automatic exclusion of non-narrative and irrelevant books

## Architecture

### Agent Flow

![Book recommender agent flow.drawio.png](Book%20recommender%20agent%20flow.drawio.png)

### Conversation Flow

1. **Initialization**: User Proxy sends `TASK:` with user's query
2. **Stage 1 - Search**: Librarian searches Google Books (10 results)
3. **Stage 1 - Screen**: Critic reviews snippets, identifies 5 promising books
4. **Stage 2 - Fetch**: Librarian gets detailed info for selected books
5. **Stage 2 - Evaluate**: Critic scores and validates, selects top 3
6. **Termination**: Critic approves with `OK: RESULT:` or requests retry

### Speaker Selection Logic

The `speaker_selection` function in `main.py` orchestrates agent turns:

- **User Proxy → Librarian**: After TASK initiation
- **Librarian → User Proxy**: When making tool calls
- **User Proxy → Critic**: After tool execution (search results)
- **Critic → User Proxy**: When scoring books
- **Critic → Librarian**: When requesting new search
- **Critic → None**: When approving results (terminates)

## Quick Start

### Prerequisites

- Python 3.8+
- API Keys:
  - Google Gemini API key
  - Google Books API key

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ML_Exam_book_recommender
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```
### Key Dependencies

- `pyautogen~=0.2.22` - Multi-agent framework
- `streamlit~=1.41.1` - Web interface
- `google-generativeai~=0.8.3` - Google Gemini API
- `sentence-transformers~=3.3.1` - Semantic scoring
- `python-dotenv~=1.0.1` - Environment configuration

See `requirements.txt` for complete list.

3. **Configure API keys** - See [Configuration](#configuration) section below

### Running the Application

#### Option 1: Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```
Access at: **http://localhost:8501**

#### Option 2: CLI Mode

```bash
# Use default provider (Google)
python main.py

# Specify LLM provider
python main.py --google
python main.py --mistral
python main.py --cerebras
```

## Configuration

### API Keys

Create a `.env` file in the project root:

```env
# Required
GOOGLE_LLM_API_KEY=your_google_api_key_here
GOOGLE_BOOKS_API_KEY=your_google_books_key_here

# Optional - for alternative LLM providers
MISTRAL_API_KEY=your_mistral_key_here
CEREBRAS_API_KEY=your_cerebras_key_here
```
### Obtaining API Keys

**Google Gemini API**:
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in and create a new API key

**Google Books API**:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Books API
3. Create credentials → API key


### LLM Configuration

Edit `config.py` or use command-line arguments to change the LLM provider:

```python
LLM_CONFIG = {
    "model": "gemini-2.0-flash",
    "api_type": "google",
    "api_key": GOOGLE_LLM_API_KEY,
    "api_rate_limit": 0.1,
    "max_retries": 3,
    "timeout": 30,
}
```

### Agent Parameters

**Librarian Agent**:
- Max search results: 10 (always)
- Max detail fetches: 5 per turn
- Semantic scoring: Automatic with all-MiniLM-L6-v2

**Internal Critic**:
- Scoring tool: `score_books_by_relevance`
- Relevance thresholds:
  - ≥80: HIGHLY relevant
  - 65-79: MODERATELY relevant
  - 50-64: SOMEWHAT relevant
  - <50: LOW relevance

**Conversation**:
- Max rounds: 20
- Termination: When critic approves with `OK:`

## Usage

### Example Queries

**Simple query**:
```
Find fantasy books about dragons published after 2015
```

**Detailed query**:
```
Find books that satisfy:
1) About artificial intelligence and ethics
2) Subject/genre is science fiction
3) Published between 2018-2024
```

**Specific author**:
```
Find books by N.K. Jemisin about world-building and magic systems
```

### Web Interface Features

- **Real-time Processing**: Watch agents collaborate in real-time
- **Conversation History**: View full agent dialogue
- **Result Formatting**: Clean presentation of top 3 recommendations
- **Search History**: Track previous queries
- **Error Handling**: Clear error messages and fallback strategies

### CLI Output

Results are saved to `logs/` directory with timestamps, including:
- Task/query
- Librarian results
- Critic evaluation
- Full chat history
- Error information (if any)

### Manual Testing

For testing purposes, edit the user prompt in `utils/task_prompts.py`, then run:

```bash
python main.py
```

Results are saved to `logs/` with timestamps.

## How It Works

### Stage 1: Initial Search & Screening

**Librarian**:
1. Parses user request (subject, author, year, keywords)
2. Calls `search_google_books(max_results=10)`
3. Applies semantic scoring to all results
4. Returns numbered list with basic info

**Critic**:
1. Reviews titles, snippets, subjects
2. Applies automatic exclusions (instruction books, non-fiction)
3. Identifies 5 most promising candidates
4. Responds with:
   - `FETCH DETAILS FOR: #1, #3, #5, #7, #9` → Proceed to Stage 2
   - `NONE PROMISING` + search suggestions → Retry with new query

### Stage 2: Detailed Fetching & Selection

**Librarian**:
1. Extracts volume IDs from Stage 1 URLs
2. Calls `get_book_details_google(volume_id)` for 5 books
3. Handles API failures with fallback strategies
4. Returns full descriptions (or reports failures)

**Critic**:
1. Calls `score_books_by_relevance` (semantic similarity)
2. Evaluates descriptions against requirements
3. Applies strict filtering
4. Responds with:
   - `OK: RESULT: TOP 3 SELECTED BOOKS` → Success (terminates)
   - `CRITIQUE:` + new suggestions → Back to Stage 1

## Error Handling

### API Failure Recovery

1. **Partial Results**: Returns successful fetches, reports failures
2. **Complete Failure**: Falls back to Stage 1 snippets
3. **Validation**: Checks volume ID/title mismatches
4. **Retry Limit**: Max 2 consecutive failures → use Stage 1 results

### RECITATION Detection

Google Gemini may block copyrighted content. The system:
- Detects "RECITATION" errors in responses
- Automatically falls back to Stage 1 snippets
- Prevents infinite retry loops

### Mismatch Detection

Validates that fetched book details match search results:
- Volume ID verification
- Title cross-reference
- Reports discrepancies to critic


## Semantic Scoring

Uses **sentence-transformers** with `all-MiniLM-L6-v2` model:

- **Input**: User query + book metadata (title, description, authors, subjects)
- **Output**: Relevance score (0-100)
- **Process**:
  1. Encode query as embedding
  2. Encode book text as embedding
  3. Calculate cosine similarity
  4. Scale to 0-100 range

**Alternative Model**: Can upgrade to `all-mpnet-base-v2` for higher accuracy (420MB vs 80MB)


