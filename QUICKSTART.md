# 🚀 Quick Start Guide

## Running the Book Recommender Web Interface

### Option 1: Double-Click (Windows)
Simply double-click `start_frontend.bat` to launch the web interface.

### Option 2: Command Line
```bash
streamlit run streamlit_app.py
```

### Option 3: PowerShell
```powershell
cd C:\Users\madsp\Documents\GitHub\ML_Exam_book_recommender
streamlit run streamlit_app.py
```

## Access the Interface
Once started, open your browser and go to:
**http://localhost:8501**

## What You'll See
- 📝 Text area to enter your book search query
- 🚀 Search button to start the AI agents
- 📚 Results section with book recommendations
- ✅ Quality evaluation from the internal critic
- 💬 Expandable sections to view agent conversations and logs

## Example Query
Try this to get started:
```
Find books that satisfy:
1) About artificial intelligence and machine learning
2) Subject/genre is computer science or technology
3) Published after 2015
4) Return top 5 books with title, authors, publication year, and URL
```

## Troubleshooting

### Port Already in Use
If you see "Address already in use", another Streamlit app is running:
```bash
# Kill existing Streamlit processes
taskkill /F /IM streamlit.exe
```

Or specify a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Docker Not Running
Make sure Docker Desktop is installed and running before starting the app.

### API Key Errors
Check your `.env` file has the required keys:
- `GOOGLE_LLM_API_KEY`
- `GOOGLE_BOOKS_API_KEY`
- `BIG_BOOK_API_KEY`

## Stopping the Server
Press `Ctrl + C` in the terminal to stop the Streamlit server.

---

For detailed documentation, see [README_FRONTEND.md](README_FRONTEND.md)

