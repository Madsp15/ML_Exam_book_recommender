import os
import sys
import streamlit as st
import logging
from io import StringIO
import dotenv
from agents.internal_critic_agent import get_internal_critic_agent
from agents.librarian_agent import get_librarian_agent_api_agent
from agents.user_proxy_agent import get_user_proxy
from autogen.coding import DockerCommandLineCodeExecutor
from utils.utils import get_llm_config, get_work_dir
from main import process_single_task

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Load environment variables
dotenv.load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Book Recommender Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        margin: 2rem 0;
        color: #C62828;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .agent-message {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
    }
    .librarian-msg {
        background-color: #E3F2FD;
        border-left: 3px solid #2196F3;
    }
    .critic-msg {
        background-color: #E8F5E9;
        border-left: 3px solid #4CAF50;
    }
    .user-msg {
        background-color: #FFF3E0;
        border-left: 3px solid #FF9800;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None

# Sidebar configuration
with st.sidebar:
    st.markdown("### Configuration")

    llm_provider = st.selectbox(
        "LLM Provider",
        options=["google", "cerebras"],
        index=0,
        help="Select the language model provider to use"
    )

    st.markdown("---")
    st.markdown("### Requirements")
    st.markdown("""
    **Before using:**
    - Set API keys in `.env` file
    - Docker must be running
    - All dependencies installed
    
    **Example prompt:**
    ```
    Find books that satisfy:
    1) About space exploration
    2) Science fiction genre
    3) Published after 2010
    4) Return top 5 books with 
       title, authors, year, URL
    ```
    """)

    if st.button("Clear Search History"):
        st.session_state.search_history = []
        st.session_state.current_result = None
        st.rerun()

# Main content
st.markdown('<p class="main-header"> Book Recommender Agent</p>', unsafe_allow_html=True)
st.markdown("Enter your book search query below and let the AI agents find the perfect recommendations for you!")

# Check for API key
if llm_provider == "mistral":
    env_var_name = "MISTRAL_API_KEY"
elif llm_provider == "google":
    env_var_name = "GOOGLE_LLM_API_KEY"
elif llm_provider == "cerebras":
    env_var_name = "CEREBRAS_API_KEY"

api_key = os.getenv(env_var_name)
if not api_key:
    st.markdown(f'<div class="error-box"> <strong>Error:</strong> {env_var_name} not found in environment variables. Please add it to your .env file.</div>', unsafe_allow_html=True)
    st.stop()

# Input section
st.markdown("### Enter Your Book Search Query")
user_prompt = st.text_area(
    "Describe what books you're looking for:",
    height=150,
    placeholder="Example: Find books about artificial intelligence written by experts, published after 2015, in the computer science category. Return top 3 books with title, authors, year, and URL.",
    help="Be specific about genre, topic, author, year, or any other criteria you want."
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_button = st.button("Search Books", type="primary", use_container_width=True)
with col2:
    example_button = st.button("Load Example", use_container_width=True)

if example_button:
    st.session_state.example_loaded = True
    st.rerun()

if "example_loaded" in st.session_state and st.session_state.example_loaded:
    user_prompt = """Find books that satisfy as many of the following constraints as possible:
1) A book about an old warrior coming back for one last battle.
2) Subject/genre is fantasy.
3) Return the top three books, providing for each: title, authors, publication year, URL."""
    st.session_state.example_loaded = False

# Process search
if search_button and user_prompt.strip():
    with st.spinner("AI agents are working on your request..."):
        executor = None  # Initialize executor to None
        try:
            # Setup logging to capture agent logs
            log_stream = StringIO()
            handler = logging.StreamHandler(log_stream)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            logger = logging.getLogger()
            logger.addHandler(handler)

            # Initialize agents
            llm_config = get_llm_config(llm_provider=llm_provider, api_key=api_key)

            # Check Docker
            try:
                executor = DockerCommandLineCodeExecutor(work_dir=get_work_dir())
            except Exception as docker_error:
                st.markdown(f'<div class="error-box"> <strong>Docker Error:</strong> {str(docker_error)}<br><br><strong>Troubleshooting:</strong><br>- Make sure Docker Desktop is installed and running<br>- Check if Docker daemon is accessible</div>', unsafe_allow_html=True)
                st.stop()

            librarian_agent = get_librarian_agent_api_agent(custom_llm_config=llm_config)
            internal_critic = get_internal_critic_agent(llm_config=llm_config, terminate_conversation=True)
            user_proxy = get_user_proxy(executor=executor)

            # Process the task
            result = process_single_task(
                user_prompt,
                llm_config,
                librarian_agent,
                internal_critic,
                user_proxy
            )

            # Capture logs
            logs = log_stream.getvalue()
            result["logs"] = logs

            # Remove handler
            logger.removeHandler(handler)

            # Store result
            st.session_state.current_result = result
            st.session_state.search_history.insert(0, result)

        except Exception as e:
            st.markdown(f'<div class="error-box"> <strong>Unexpected Error:</strong> {str(e)}</div>', unsafe_allow_html=True)
            st.stop()
        finally:
            # Stop the Docker executor to clean up containers
            if executor:
                executor.stop()
                logging.info("Docker executor stopped.")

# Display results
if st.session_state.current_result:
    result = st.session_state.current_result

    st.markdown("---")
    st.markdown("##  Results")

    # Check for errors
    if result.get("error"):
        st.markdown(f'<div class="error-box"> <strong>Error occurred during processing:</strong><br>{result["error"]}</div>', unsafe_allow_html=True)

        # Show error details in expander
        with st.expander(" View Error Details", expanded=False):
            st.code(result["error"], language="text")

        # Show logs if available
        if result.get("logs"):
            with st.expander(" View Processing Logs", expanded=False):
                st.text(result["logs"])
    else:
        # Display librarian results
        st.markdown("### Recommended Books")

        if result.get("librarian_result"):
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown(result["librarian_result"])
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">ℹ No results found</div>', unsafe_allow_html=True)

        # Display critic evaluation
        if result.get("critic_evaluation"):
            st.markdown("### Quality Evaluation")
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(result["critic_evaluation"])
            st.markdown('</div>', unsafe_allow_html=True)

        # Expandable sections for detailed information
        col1, col2 = st.columns(2)

        with col1:
            with st.expander("View Agent Conversation", expanded=False):
                if result.get("chat_history"):
                    for msg in result["chat_history"]:
                        agent_name = msg.get("name", "Unknown")
                        content = msg.get("content", "")
                        role = msg.get("role", "")

                        # Determine styling based on agent
                        if "librarian" in agent_name.lower():
                            css_class = "librarian-msg"
                            icon = "(づ ᴗ _ᴗ)づ 📖"
                        elif "critic" in agent_name.lower():
                            css_class = "critic-msg"
                            icon = "☝️🤓"
                        else:
                            css_class = "user-msg"
                            icon = "🧍‍♂️"

                        if content and content.strip():
                            st.markdown(f'<div class="agent-message {css_class}"><strong>{icon} {agent_name}</strong></div>', unsafe_allow_html=True)
                            st.text(content[:500] + "..." if len(content) > 500 else content)
                            st.markdown("---")
                else:
                    st.info("No conversation history available")

        with col2:
            with st.expander(" View Processing Logs", expanded=False):
                if result.get("logs"):
                    st.text(result["logs"])
                else:
                    st.info("No logs available")

# Search history
if st.session_state.search_history:
    st.markdown("---")
    st.markdown("##  Search History")

    for idx, hist_result in enumerate(st.session_state.search_history[:5]):  # Show last 5
        with st.expander(f"Search {idx + 1}: {hist_result['task'][:80]}...", expanded=False):
            if hist_result.get("error"):
                st.error(f"Error: {hist_result['error']}")
            else:
                st.markdown("**Results:**")
                st.text(hist_result.get("librarian_result", "No results")[:300] + "...")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p> Book Recommender Agent - Powered by AutoGen & Multi-Agent AI</p>
    <p>Made with  using Streamlit</p>
</div>
""", unsafe_allow_html=True)

