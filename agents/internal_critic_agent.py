from autogen import AssistantAgent

from utils.utils import EVALUATION_CRITERIA


def get_system_message(terminate_conversation: bool) -> str:
    base_message = f"""
        You are an internal critic who evaluates the relevance of answers provided by a research paper search agent in response to user queries.
        Your task is to critically assess whether the latest answer from the search agent adequately addresses the user's research task,
        and if it fully meets the specified constraints. When the agent states it cannot fulfill the task, evaluate whether this is a reasonable conclusion based on the constraints given.

        {EVALUATION_CRITERIA}

        Rules:
        - If the latest answer is acceptable (either a solution, or an acceptance of the agent's limitations) respond with:
          OK: <short justification>
          RESULT: <the answer from the search agent>
        - If there are issues, respond with:
          CRITIQUE: <what is wrong + smallest fix needed>
        - Do NOT propose your own final answer; only judge and comment.
        """
    if terminate_conversation:
        base_message += "\n - After providing an OK and RESULT, end the conversation with TERMINATE."
    return base_message


def get_internal_critic_agent(
    llm_config: dict, terminate_conversation: bool = False
) -> AssistantAgent:
    internal_critic = AssistantAgent(
        name="internal_critic",
        llm_config=llm_config,
        system_message=get_system_message(
            terminate_conversation=terminate_conversation
        ),
    )

    return internal_critic
