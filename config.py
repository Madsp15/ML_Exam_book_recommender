"""
Configuration file for the book recommender system
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM Configuration
LLM_CONFIG = {
"model": "gemini-2.0-flash",
"api_type": "google",
"api_key": GOOGLE_API_KEY,
"api_rate_limit": 0.1,
"max_retries": 3,
"num_predict": -1,
"repeat_penalty": 1.1,
"native_tool_calls": False,
"stream": False,
"seed": 23,
"cache_seed": None,
"timeout": 30,
}

# Open Library API Configuration
OPEN_LIBRARY_BASE_URL = "https://openlibrary.org"
OPEN_LIBRARY_SEARCH_URL = f"{OPEN_LIBRARY_BASE_URL}/search.json"
OPEN_LIBRARY_WORKS_URL = f"{OPEN_LIBRARY_BASE_URL}/works"

# Agent Configuration
MAX_CONSECUTIVE_AUTO_REPLY = 10

