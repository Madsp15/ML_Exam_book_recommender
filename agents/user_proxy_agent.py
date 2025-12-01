from autogen import UserProxyAgent
from autogen.coding import DockerCommandLineCodeExecutor

from tools import search_open_library, get_book_details, search_google_books, get_book_details_google


def get_user_proxy(executor: DockerCommandLineCodeExecutor) -> UserProxyAgent:
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        llm_config=False,
        is_termination_msg=lambda msg: (
            isinstance(msg, dict)
            and isinstance(msg.get("content"), str)
            and "TERMINATE" in msg["content"]
        ),
        code_execution_config={
            "executor": executor,
        },
    )

    user_proxy.register_for_execution(
        name="search_google_books",
    )(search_google_books)

    user_proxy.register_for_execution(
        name="get_book_details_google",
    )(get_book_details_google)

    return user_proxy
