import os
import sys
import os
import dotenv
import logging
import argparse
from agents.internal_critic_agent import get_internal_critic_agent
from agents.librarian_agent import get_librarian_agent_api_agent, set_current_query

from agents.user_proxy_agent import get_user_proxy

from utils.task_prompts import simple_tasks
from autogen import (
    GroupChat,
    GroupChatManager,
)
from utils.utils import (
    extract_final_answer,
    save_results,
    get_llm_config,
)



logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO
)

# Set UTF-8 encoding for Windows console to handle Unicode characters
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


# Module-level initialization is now done only when running as main script
# to avoid issues when importing from streamlit_app.py


def has_critic_approval(messages: list[dict]) -> bool:
    """Check if internal critic has approved the result with OK:"""
    return any(
        msg.get("name") == "internal_critic" and "OK:" in (msg.get("content") or "")
        for msg in messages
    )


def speaker_selection(last_speaker, groupchat):
    messages = groupchat.messages
    last_message = messages[-1] if messages else {}
    last_message_content = last_message.get("content", "") if messages else ""

    # Get agents from the groupchat to avoid referencing global variables
    agents_dict = {agent.name: agent for agent in groupchat.agents}
    librarian = agents_dict.get("librarianAgentApiAgent")
    critic = agents_dict.get("internal_critic")
    proxy = agents_dict.get("user_proxy")

    # Check for RECITATION errors (Google Gemini copyright detection)
    # Look for "Unsuccessful Finish Reason: RECITATION" or "MALFORMED_FUNCTION_CALL" in recent messages
    recitation_detected = False
    malformed_call_detected = False

    for msg in messages[-5:]:  # Check last 5 messages for patterns
        content = msg.get("content", "")
        if "RECITATION" in content or "Unsuccessful Finish Reason" in content:
            recitation_detected = True
            logging.warning("RECITATION error detected - copyright content blocked by Gemini")
            break
        if "MALFORMED_FUNCTION_CALL" in content or "NO CONTENT RETURNED" in content:
            malformed_call_detected = True
            logging.warning("Malformed function call detected - likely due to content policy")
            break

    if recitation_detected or malformed_call_detected:
        # Force fallback to Stage 1 results immediately
        # Inject a message to the critic to use Stage 1 results
        logging.info("Triggering fallback to Stage 1 search results due to content policy issues")
        # The critic should detect this and use Stage 1 results
        return critic

    # kick-off: user_proxy starts with TASK, librarian agent responds
    if last_speaker is proxy and last_message_content.strip().startswith("TASK:"):
        return librarian

    # After user_proxy speaks (e.g., tool execution result)
    if last_speaker is proxy:
        # If it's a tool result, route back to whoever called the tool
        if last_message.get("role") == "tool":
            # Look back to find who made the tool call
            for msg in reversed(messages[:-1]):  # Skip the current message
                if "tool_calls" in msg:
                    caller_name = msg.get("name")
                    if caller_name == "internal_critic":
                        return critic
                    elif caller_name == "librarianAgentApiAgent":
                        return librarian
                    break
            # Default: return to librarian if we can't determine the caller
            return librarian
        else:
            return critic

    # After librarian speaks
    if last_speaker is librarian:
        # If librarian made a tool call, user_proxy executes it
        if "tool_calls" in last_message:
            return proxy
        else:
            # No tool call, librarian returned filtered books - send to critic
            return critic

    # After critic speaks
    if last_speaker is critic:
        # If critic made a tool call, user_proxy executes it
        if "tool_calls" in last_message:
            return proxy
        elif "OK:" in last_message_content:
            # Critic approved, conversation can end
            return None
        else:
            # Critic wants changes, send back to librarian
            return librarian

    # Default fallback
    return librarian


def process_single_task(task: str, llm_config: dict, librarian_agent, internal_critic, user_proxy):
    """
    Process a single task through the agent system.

    Args:
        task: The book search task/prompt
        llm_config: LLM configuration dict
        librarian_agent: The librarian agent instance
        internal_critic: The internal critic agent instance
        user_proxy: The user proxy agent instance

    Returns:
        dict with keys: task, librarian_result, critic_evaluation, chat_history, error
    """
    def should_terminate(msg) -> bool:
        # Terminate when internal critic has approved with OK:
        return has_critic_approval(group.messages)

    group = GroupChat(
        agents=[user_proxy, librarian_agent, internal_critic],
        max_round=20,
        speaker_selection_method=speaker_selection,
        allow_repeat_speaker=False,
    )

    manager = GroupChatManager(
        name="main_manager",
        groupchat=group,
        llm_config=llm_config,
        is_termination_msg=should_terminate,
    )

    logging.info(f"\n{'='*80}\nStarting task: {task}\n{'='*80}")

    # Set current query for semantic scoring
    set_current_query(task)

    try:
        chat = user_proxy.initiate_chat(
            manager,
            message=f"TASK: {task}",
            max_turns=20,
            summary_method="reflection_with_llm",
        )

        # Extract critic's evaluation first to determine result source
        critic_evaluation = None
        result_source = "librarian"  # Track where the result came from
        for msg in reversed(chat.chat_history):
            if msg.get("name") == "internal_critic" and "OK:" in (msg.get("content") or ""):
                critic_evaluation = msg.get("content")
                # Check if result came from critic's fallback (API failure scenario)
                if "Using Stage 1 search results" in critic_evaluation or "API detail fetching failed" in critic_evaluation:
                    result_source = "critic_fallback"
                break

        # Extract the final approved result
        # For fallback scenarios, use critic's RESULT directly
        # For normal flow, use librarian's RESULT
        if result_source == "critic_fallback":
            # Extract RESULT from critic's message
            librarian_result = extract_final_answer(chat, "internal_critic")
        else:
            # Normal flow: check librarian first, then fall back to critic
            librarian_result = extract_final_answer(chat, "librarianAgentApiAgent")

        logging.info(f"\n{'='*80}\nFinal Result (source: {result_source}):\n{librarian_result}\n")
        logging.info(f"Critic Evaluation:\n{critic_evaluation}\n{'='*80}\n")

        return {
            "task": task,
            "librarian_result": librarian_result,
            "critic_evaluation": critic_evaluation,
            "chat_history": chat.chat_history,
            "result_source": result_source,  # 'librarian' or 'critic_fallback'
            "error": None,
        }
    except TypeError as e:
        error_msg = f"Gemini API error (TypeError): {str(e)}"
        logging.error(f"\n{'='*80}\nError processing task: {error_msg}\n{'='*80}\n")
        return {
            "task": task,
            "librarian_result": None,
            "critic_evaluation": None,
            "chat_history": None,
            "error": error_msg,
        }
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
        logging.error(f"\n{'='*80}\nError processing task: {error_msg}\n{'='*80}\n")
        return {
            "task": task,
            "librarian_result": None,
            "critic_evaluation": None,
            "chat_history": None,
            "error": error_msg,
        }


def main():
    """Main function to run the book recommender agent."""
    # Load environment variables
    dotenv.load_dotenv()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Book Recommender Agent")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--llm-provider",
        choices=["mistral", "google", "cerebras"],
        dest="llm_provider",
        default="google",
    )
    group.add_argument(
        "--google",
        action="store_const",
        const="google",
        dest="llm_provider",
        help="Use Google as LLM provider",
    )
    group.add_argument(
        "--mistral",
        action="store_const",
        const="mistral",
        dest="llm_provider",
        help="Use Mistral as LLM provider",
    )
    group.add_argument(
        "--cerebras",
        action="store_const",
        const="cerebras",
        dest="llm_provider",
        help="Use Cerebras as LLM provider",
    )

    args = parser.parse_args()
    llm_provider = args.llm_provider

    if llm_provider == "mistral":
        env_var_name = "MISTRAL_API_KEY"
    elif llm_provider == "google":
        env_var_name = "GOOGLE_LLM_API_KEY"
    elif llm_provider == "cerebras":
        env_var_name = "CEREBRAS_API_KEY"
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    api_key = os.getenv(env_var_name)
    if not api_key:
        raise ValueError(f"{env_var_name} not found in environment variables.")

    LLM_CONFIG = get_llm_config(llm_provider=llm_provider, api_key=api_key)

    librarian_agent = get_librarian_agent_api_agent(custom_llm_config=LLM_CONFIG)
    internal_critic = get_internal_critic_agent(llm_config=LLM_CONFIG, terminate_conversation=True)
    user_proxy = get_user_proxy()

    try:
        results = []
        for task in simple_tasks:
            result = process_single_task(
                task, LLM_CONFIG, librarian_agent, internal_critic, user_proxy
            )
            results.append(result)

        save_results(results)
    except Exception as e:
        logging.error(f"Fatal error in main execution: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    main()