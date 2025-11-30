import os
import sys
import dotenv
import logging
import argparse
from agents.internal_critic_agent import get_internal_critic_agent
from agents.librarian_agent import get_librarian_agent_api_agent

from agents.user_proxy_agent import get_user_proxy

from utils.task_prompts import simple_tasks
from autogen.coding import DockerCommandLineCodeExecutor
from autogen import (
    GroupChat,
    GroupChatManager,
)
from utils.utils import (
    extract_final_answer,
    save_results,
    get_llm_config,
    get_work_dir,
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


dotenv.load_dotenv()
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

executor = DockerCommandLineCodeExecutor(
    work_dir=get_work_dir(),
)

librarian_agent = get_librarian_agent_api_agent(custom_llm_config=LLM_CONFIG)
internal_critic = get_internal_critic_agent(llm_config=LLM_CONFIG, terminate_conversation=True)
user_proxy = get_user_proxy(executor=executor)


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

    # kick-off: user_proxy starts with TASK, librarian agent responds
    if last_speaker is user_proxy and last_message_content.strip().startswith("TASK:"):
        return librarian_agent

    # After user_proxy speaks (e.g., tool execution result)
    if last_speaker is user_proxy:
        # If it's a tool result, librarian should process it
        if last_message.get("role") == "tool":
            return librarian_agent
        else:
            return internal_critic

    # After librarian speaks
    if last_speaker is librarian_agent:
        # If librarian made a tool call, user_proxy executes it
        if "tool_calls" in last_message:
            return user_proxy
        else:
            # No tool call, librarian returned filtered books - send to critic
            return internal_critic

    # After critic speaks
    if last_speaker is internal_critic:
        if "OK:" in last_message_content:
            # Critic approved, conversation can end
            return None
        else:
            # Critic wants changes, send back to librarian
            return librarian_agent

    # Default fallback
    return librarian_agent


def main():
    def should_terminate(msg) -> bool:
        # Terminate when internal critic has approved with OK:
        return has_critic_approval(group.messages)

    group = GroupChat(
        agents=[user_proxy, librarian_agent, internal_critic],
        max_round=20,  # Increased to allow multiple tool calls by librarian
        speaker_selection_method=speaker_selection,
        allow_repeat_speaker=False,
    )

    manager = GroupChatManager(
        name="main_manager",
        groupchat=group,
        llm_config=LLM_CONFIG,
        is_termination_msg=should_terminate,
    )

    results = []
    for task in simple_tasks:
        logging.info(f"\n{'='*80}\nStarting task: {task}\n{'='*80}")

        chat = user_proxy.initiate_chat(
            manager,
            message=f"TASK: {task}",
            max_turns=20,  # Increased to allow for multiple tool calls
            summary_method="reflection_with_llm",
        )

        # Extract the final approved result from internal critic
        librarian_result = extract_final_answer(chat, "librarianAgentApiAgent")

        # Extract critic's evaluation
        critic_evaluation = None
        for msg in reversed(chat.chat_history):
            if msg.get("name") == "internal_critic" and "OK:" in (msg.get("content") or ""):
                critic_evaluation = msg.get("content")
                break

        logging.info(f"\n{'='*80}\nLibrarian Result:\n{librarian_result}\n")
        logging.info(f"Critic Evaluation:\n{critic_evaluation}\n{'='*80}\n")

        results.append(
            {
                "task": task,
                "librarian_result": librarian_result,
                "critic_evaluation": critic_evaluation,
            }
        )

    save_results(results)


if __name__ == "__main__":
    main()