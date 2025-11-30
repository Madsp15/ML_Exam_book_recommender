import json
from typing import Dict, List
from autogen import AssistantAgent

from utils.utils import EVALUATION_CRITERIA


def build_external_judge_prompt(task: str, results: dict[str, str]) -> str:
    return f"""
        You are evaluating book search agents' results for the following task:
        "{task}"

        Here are the results from different agents:
        {results}

        Evaluate accordingly to your system message and return the required JSON scores.
        """

def llm_judge_score(
        judge_agent: AssistantAgent, user_prompt: str, results: dict[str, str]
) -> Dict:
    judge_prompt = build_external_judge_prompt(user_prompt, results)
    raw = judge_agent.generate_reply(
        messages=[{"role": "user", "content": judge_prompt}]
    )
    content = (
        raw["content"]
        .replace("TERMINATE:", "")
        .replace("```json", "")
        .replace("```", "")
        .replace("\n", "")
        .strip()
    )

    return json.loads(content)


def get_external_judge_agent(custom_llm_config: dict) -> AssistantAgent:
    judge = AssistantAgent(
        name="judge",
        llm_config=custom_llm_config,
        system_message=f"""
                   You are an external evaluator of book search agents. You will be given several sets of results from different agents attempting to complete the same book search task.
                   Your role is to score and compare the quality of the results based on how well they meet the task requirements.

                   {EVALUATION_CRITERIA}

                   Return the response exactly as follows (expand the array according to the number of results), no extra commentary:
                   TERMINATE:
                   ```json
                   {{
                       "result_1": {{
                           "completeness": <score 1-5>,
                           "relevance": <score 1-5>,
                           "accuracy": <score 1-5>,
                           "usefulness": <score 1-5>
                       }},
                       "result_2": {{
                           "completeness": <score 1-5>,
                           "relevance": <score 1-5>,
                           "accuracy": <score 1-5>,
                           "usefulness": <score 1-5>
                       }}
                   }}
                   ```
               """,
    )

    return judge
