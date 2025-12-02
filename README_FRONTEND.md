# Book Recommender Frontend

A web-based interface for the Book Recommender Agent system, built with Streamlit.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Docker Desktop** installed and running
3. **API Keys** configured in `.env` file

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your `.env` file with required API keys:
```env
GOOGLE_LLM_API_KEY=your_google_api_key_here
GOOGLE_BOOKS_API_KEY=your_google_books_key_here
BIG_BOOK_API_KEY=your_big_book_api_key_here

# Optional: if using other LLM providers
MISTRAL_API_KEY=your_mistral_key_here
CEREBRAS_API_KEY=your_cerebras_key_here
```

### Running the Frontend

1. Make sure Docker Desktop is running

2. Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

3. Your browser should automatically open to `http://localhost:8501`

If it doesn't open automatically, navigate to: **http://localhost:8501**

## 📖 How to Use

### Basic Usage

1. **Enter Your Query**: Type your book search requirements in the text area
   - Be specific about genre, topic, author, publication year, etc.
   - Specify how many results you want
   
2. **Click "Search Books"**: The AI agents will work together to find recommendations

3. **View Results**: See the recommended books with titles, authors, years, and URLs

### Example Queries

**Fantasy Adventure:**
```
Find books that satisfy:
1) A book about an old warrior coming back for one last battle
2) Subject/genre is fantasy
3) Return the top three books with title, authors, publication year, URL
```

**Science Fiction:**
```
Find science fiction books about:
1) Artificial intelligence and consciousness
2) Published after 2010
3) Written by well-known authors
4) Return top 5 books with full details
```

**Historical Fiction:**
```
Find historical fiction books:
1) Set during World War II
2) From a female author's perspective
3) Published in the last 20 years
4) Return 4 books with title, author, year, and description
```

## 🔍 Features

### Main Interface
- **Text Input**: Large text area for detailed search queries
- **LLM Provider Selection**: Choose between Google, Mistral, or Cerebras
- **Load Example**: Quick-load an example query to get started

### Results Display
- **Book Recommendations**: Formatted list with all requested details
- **Quality Evaluation**: Internal critic's assessment of the results
- **Expandable Logs**: Click to view detailed agent conversation
- **Error Display**: Clear error messages with troubleshooting tips

### Session Features
- **Search History**: View your last 5 searches
- **Clear History**: Reset your session
- **Persistent Results**: Results stay visible until new search

## 🛠️ Troubleshooting

### "API Key not found" Error
- Check that your `.env` file exists in the project root
- Verify the API key name matches: `GOOGLE_LLM_API_KEY`, `MISTRAL_API_KEY`, or `CEREBRAS_API_KEY`
- Restart the Streamlit app after adding keys

### "Docker Error" Message
- **Install Docker Desktop**: Download from https://www.docker.com/products/docker-desktop
- **Start Docker**: Make sure Docker Desktop is running before launching the app
- **Check Docker Status**: Run `docker ps` in terminal to verify Docker is working

### "No Results Found"
- Try making your query more specific
- Check the agent conversation logs to see what searches were attempted
- Verify the API keys for Google Books are valid

### Slow Response Times
- First searches are slower as Docker containers initialize
- Complex queries with multiple criteria take longer to process
- Check your internet connection for API calls

## 📋 Advanced Features

### Viewing Agent Logs

The frontend captures detailed logs from the agent conversations:

1. **Agent Conversation Tab**: 
   - See the full back-and-forth between librarian and critic agents
   - Color-coded by agent type (blue for librarian, green for critic)
   - Shows tool calls and API responses

2. **Processing Logs Tab**:
   - View technical logs with timestamps
   - Useful for debugging issues
   - Shows API calls and internal processing

### Search History

- Automatically saves your last 5 searches
- Click on previous searches to see summarized results
- Use "Clear Search History" to reset

### LLM Provider Options

**Google (Default)**
- Model: Gemini 2.0 Flash
- Fast and reliable
- Best for general use

**Mistral**
- Alternative provider
- Requires MISTRAL_API_KEY

**Cerebras**
- High-performance option
- Requires CEREBRAS_API_KEY

## 🎨 Interface Customization

The app uses custom CSS styling with:
- Color-coded message boxes (success in green, errors in red, info in blue)
- Responsive layout that works on different screen sizes
- Professional typography and spacing

## 💡 Tips

1. **Be Specific**: The more detailed your query, the better the results
2. **Use Numbered Lists**: Helps the agents parse your requirements
3. **Specify Quantity**: Always mention how many books you want (top 3, top 5, etc.)
4. **Check Logs**: If results seem off, expand the conversation logs to understand why
5. **Try Examples**: Use the "Load Example" button to see query formatting

## 🔄 Comparing with CLI Version

**Frontend Advantages:**
- No need to edit Python files
- Real-time feedback and progress indicators
- Easy to try multiple queries
- Visual formatting of results
- Click-to-expand logs instead of terminal scrolling

**CLI Version (main.py):**
- Better for batch processing multiple queries
- Saves results to JSON files automatically
- Useful for automated testing
- No browser required

## 📝 Notes

- Each search creates a new agent conversation
- Results are stored in session state (not persisted to disk)
- Docker containers are reused between searches for efficiency
- API rate limits from your LLM provider still apply

## 🆘 Support

If you encounter issues:

1. Check the error messages in the red boxes
2. Expand the "View Processing Logs" section
3. Verify all prerequisites are installed
4. Check that `.env` file is properly configured
5. Ensure Docker is running and accessible

---

**Happy Book Hunting! 📚**

