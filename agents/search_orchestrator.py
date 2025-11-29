import logging
from typing import Any, override
from autogen import (
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
)
from autogen.coding import DockerCommandLineCodeExecutor

from agents.internal_critic_agent import get_internal_critic_agent
from agents.user_proxy_agent import get_user_proxy


class SearchOrchestrator(ConversableAgent):
    def __init__(
        self,
        custom_llm_config: dict,
        search_agent: AssistantAgent,
        executor: DockerCommandLineCodeExecutor,
        terminate_conversation: bool = False,
        max_turns: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.user_proxy = get_user_proxy(executor)
        self.critic = get_internal_critic_agent(
            custom_llm_config, terminate_conversation=terminate_conversation
        )
        self.search_agent = search_agent
        self.group = GroupChat(
            agents=[self.search_agent, self.critic, self.user_proxy],
            max_round=max_turns,
            speaker_selection_method=self.speaker_selection,
            allow_repeat_speaker=False,
        )
        self.group_manager = GroupChatManager(
            name=self.name + "_group_manager",
            groupchat=self.group,
            llm_config=custom_llm_config,
            is_termination_msg=lambda msg: (
                isinstance(msg, dict)
                and isinstance(msg.get("content"), str)
                and "OK:" in msg["content"]
            ),
        )

    def speaker_selection(
        self, last_speaker, group: GroupChat
    ) -> ConversableAgent | None:
        messages = group.messages
        last_message = messages[-1] if messages else {}
        last_message_content = last_message.get("content", "") if messages else ""

        # After user_proxy speaks
        if last_speaker is self.user_proxy:
            # If it's the initial TASK message or a tool result, search agent should respond
            if last_message_content.strip().startswith("TASK:"):
                return self.search_agent
            elif last_message.get("role") == "tool":
                return self.search_agent
            else:
                # User proxy shouldn't speak in other cases in this flow
                return self.critic

        # After search agent speaks
        elif last_speaker is self.search_agent:
            # Check if the search agent made a tool call
            if "tool_calls" in last_message:
                # If there's a tool call, let user_proxy execute it
                return self.user_proxy
            else:
                # If no tool call (just a response), send to critic for evaluation
                return self.critic

        # After critic speaks
        elif last_speaker is self.critic:
            if "OK:" in last_message_content:
                # Critic approved, end the conversation
                return None
            else:
                # Critic wants changes, send back to search agent
                return self.search_agent

        # Default fallback
        return self.search_agent

    @override
    def generate_reply(
        self, messages=None, sender=None, exclude=None
    ) -> str | dict[str, Any] | None:
        task = "No task provided."

        # First: Try to get messages from the outer groupchat (via sender)
        if sender is not None and hasattr(sender, "groupchat"):
            outer_messages = sender.groupchat.messages
            for msg in outer_messages:
                content = msg.get("content", "") or ""
                if "TASK:" in content:
                    task = content
                    break

        # Fallback: check the messages passed directly
        if task == "No task provided." and messages:
            for msg in reversed(messages):
                if isinstance(msg, str):
                    content = msg
                else:
                    content = msg.get("content", "") or ""
                if "TASK:" in content:
                    task = content
                    break

        # Clear any previous messages in the group chat before starting fresh
        self.group.messages.clear()

        try:
            self.user_proxy.initiate_chat(
                self.group_manager,
                message=task,
                max_turns=5,
                summary_method="reflection_with_llm",
            )
        except Exception as e:
            logging.error(f"Error during inner conversation: {e}")
            return f"Error during inner conversation. {e}"

        # Get the final winning solution from the judge
        final_messages = self.group_manager.groupchat.messages
        winning_solution = None

        for msg in reversed(final_messages):
            content = msg.get("content", "")
            if content.startswith("OK:"):
                winning_solution = content
                break

        return winning_solution or "No winning solution found."
