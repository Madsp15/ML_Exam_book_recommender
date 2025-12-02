"""
Utility functions for handling Mistral API's strict message ordering requirements.
Mistral API doesn't allow 'user' role messages immediately after 'tool' role messages.
"""
from typing import List, Dict, Any


def transform_messages_for_mistral(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform messages to comply with Mistral's message ordering requirements.

    Mistral requires: user -> assistant -> [tool] -> assistant (not user)

    This function converts any 'user' role message that follows a 'tool' role message
    into an 'assistant' role message to satisfy Mistral's API requirements.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        Transformed list of messages compatible with Mistral API
    """
    if not messages:
        return messages

    transformed = []
    for i, msg in enumerate(messages):
        # Check if previous message was 'tool' and current is 'user'
        if i > 0 and transformed[-1].get("role") == "tool" and msg.get("role") == "user":
            # Convert user to assistant after tool
            msg_copy = msg.copy()
            msg_copy["role"] = "assistant"
            transformed.append(msg_copy)
        else:
            transformed.append(msg)

    return transformed

