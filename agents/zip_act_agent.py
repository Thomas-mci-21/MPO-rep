import logging
import os
import re
import json
from typing import List, Dict, Any, Tuple

import backoff
import openai
from openai import OpenAI
from colorama import Fore

from .base import BaseAgent

logger = logging.getLogger("agent_eval")

class ZipActAgent(BaseAgent):
    """
    An agent that uses a two-phase approach:
    1. StateUpdater (Memory Update): A reflective LLM call to update the state and subgoals.
    2. ZipAct (Decision): A fast-thinking LLM call to decide the next action.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # --- LLM Client Initialization ---
        # Prioritize values from the main agent config (which come from run_experiment.sh)
        # over sub-configs in zip_act.json
        main_model_name = config.get("model_name")
        main_api_base = config.get("api_base")
        main_api_key = config.get("api_key", os.environ.get("OPENAI_API_KEY"))


        zip_act_llm_config = config.get("zip_act_llm_config", {})
        self.zip_act_client = OpenAI(
            base_url=main_api_base or zip_act_llm_config.get("api_base"),
            api_key=main_api_key or zip_act_llm_config.get("api_key"),
        )
        self.zip_act_model_name = main_model_name or zip_act_llm_config.get("model_name")

        state_updater_llm_config = config.get("state_updater_llm_config", {})
        self.state_updater_client = OpenAI(
            base_url=main_api_base or state_updater_llm_config.get("api_base"),
            api_key=main_api_key or state_updater_llm_config.get("api_key"),
        )
        self.state_updater_model_name = main_model_name or state_updater_llm_config.get("model_name")


        # --- Memory and ICL Initialization ---
        self.state_list: List[str] = []
        self.subgoal_list: List[str] = []
        self.is_first_turn = True
        
        self.zip_act_examples = []
        if config.get("zip_act_icl_path"):
            with open(config["zip_act_icl_path"], 'r') as f:
                self.zip_act_examples = json.load(f)

        self.state_updater_examples = []
        if config.get("state_updater_icl_path"):
            with open(config["state_updater_icl_path"], 'r') as f:
                self.state_updater_examples = json.load(f)

        self.subgoal_planner_examples = []
        if config.get("subgoal_planner_icl_path"):
            with open(config["subgoal_planner_icl_path"], 'r') as f:
                self.subgoal_planner_examples = json.load(f)

    @backoff.on_exception(
        backoff.fibo,
        (openai.APIError, openai.Timeout, openai.RateLimitError, openai.APIConnectionError),
    )
    def _call_llm(self, client: OpenAI, model_name: str, messages: List[Dict[str, str]]) -> Tuple[str, Any]:
        """A generic, decorated method to call an LLM."""
        # The diagnostic try-except has been removed. Backoff handles retries,
        # and AuthenticationError will now be caught by outer try-excepts or terminate as expected.
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=self.config.get("max_completion_tokens", 512),
            temperature=self.config.get("temperature", 0),
            stop=self.stop_words,
        )
        content = response.choices[0].message.content
        usage = response.usage
        return content, usage

    @backoff.on_exception(
        backoff.fibo,
        (openai.APIError, openai.Timeout, openai.RateLimitError, openai.APIConnectionError),
    )
    def _call_llm_json(self, client: OpenAI, model_name: str, messages: List[Dict[str, str]]) -> Tuple[str, Any]:
        """A decorated method to call an LLM with JSON mode enabled."""
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=self.config.get("max_completion_tokens", 512),
            temperature=self.config.get("temperature", 0),
            stop=self.stop_words,
        )
        content = response.choices[0].message.content
        usage = response.usage
        return content, usage

    def _construct_state_updater_prompt(self, last_action: str, new_observation: str) -> List[Dict[str, str]]:
        """Constructs the prompt for the StateUpdater LLM, including few-shot examples."""
        system_message = {
            "role": "system",
            "content": """You are a meticulous memory assistant for an agent. Your task is to update the agent's memory based on the last action and the new observation.

RULES:
1. Update based *only* on evidence from 'Action Taken' and 'New Observation'.
2. Edits should be minimal. Do not remove information unless contradicted.
3. Do not invent facts.
4. Remove a subgoal only if there is clear evidence of its completion.
5. You MUST respond with a JSON object of the format: {"new_state": ["state item 1", ...], "new_subgoals": ["subgoal item 1", ...]}"""
        }
        
        messages = [system_message]
        for example in self.state_updater_examples:
            old_state = "\n".join(f"- {s}" for s in example["old_state_list"]) if example["old_state_list"] else "None"
            old_subgoals = "\n".join(f"- {g}" for g in example["old_subgoal_list"]) if example["old_subgoal_list"] else "None"
            user_prompt = f"""PREVIOUS MEMORY:
<Current State>
{old_state}
</Current State>
<Subgoals>
{old_subgoals}
</Subgoals>

CONTEXT:
<Action Taken>
{example["last_action"]}
</Action Taken>

<New Observation>
{example["new_observation"]}
</New Observation>

TASK: Based on the context, provide the updated memory as a JSON object."""
            messages.append({"role": "user", "content": user_prompt})

            assistant_response = json.dumps({
                "new_state": example["expected_new_state_list"],
                "new_subgoals": example["expected_new_subgoal_list"]
            })
            messages.append({"role": "assistant", "content": assistant_response})

        formatted_state = "\n".join(f"- {s}" for s in self.state_list) if self.state_list else "None"
        formatted_subgoals = "\n".join(f"- {g}" for g in self.subgoal_list) if self.subgoal_list else "None"
        final_user_prompt = f"""PREVIOUS MEMORY:
<Current State>
{formatted_state}
</Current State>
<Subgoals>
{formatted_subgoals}
</Subgoals>

CONTEXT:
<Action Taken>
{last_action}
</Action Taken>

<New Observation>
{new_observation}
</New Observation>

TASK: Based on the context, provide the updated memory as a JSON object."""
        messages.append({"role": "user", "content": final_user_prompt})
        
        return messages

    def _parse_new_memory(self, llm_output: str) -> Tuple[List[str], List[str]]:
        """Parses the StateUpdater LLM's JSON output into state and subgoal lists."""
        try:
            data = json.loads(llm_output)
            new_state_list = data.get("new_state", self.state_list)
            new_subgoal_list = data.get("new_subgoals", self.subgoal_list)

            if not isinstance(new_state_list, list) or not isinstance(new_subgoal_list, list):
                logger.error(f"StateUpdater JSON output is malformed: 'new_state' or 'new_subgoals' is not a list.\nOutput was:\n{llm_output}")
                return self.state_list, self.subgoal_list

            logger.debug(f"StateUpdater parsed memory. New state items: {len(new_state_list)}, New subgoal items: {len(new_subgoal_list)}")
            return new_state_list, new_subgoal_list
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from StateUpdater output: {e}\nOutput was:\n{llm_output}")
            return self.state_list, self.subgoal_list
        except Exception as e:
            logger.error(f"An unexpected error occurred while parsing StateUpdater output: {e}\nOutput was:\n{llm_output}")
            return self.state_list, self.subgoal_list

    def _construct_subgoal_planner_prompt(self, task_description: str) -> List[Dict[str, str]]:
        """Constructs the prompt for the SubgoalPlanner LLM to decompose the task into subgoals."""
        system_message = {
            "role": "system",
            "content": """You are a task planning assistant. Given a high-level task description, decompose it into a sequence of actionable subgoals.

RULES:
1. Each subgoal should be clear, specific, and executable.
2. Subgoals should be ordered logically to accomplish the task.
3. Keep subgoals concise (one action per subgoal).
4. You MUST respond with a JSON object of the format: {"subgoals": ["subgoal 1", "subgoal 2", ...]}"""
        }
        
        messages = [system_message]
        for example in self.subgoal_planner_examples:
            user_prompt = f"Task: {example['task']}"
            messages.append({"role": "user", "content": user_prompt})
            
            # Format the assistant's response as a JSON string
            assistant_response = json.dumps({"subgoals": example["expected_subgoals"]})
            messages.append({"role": "assistant", "content": assistant_response})

        final_user_prompt = f"Task: {task_description}"
        messages.append({"role": "user", "content": final_user_prompt})
        
        return messages

    def _parse_subgoal_planner_response(self, llm_output: str) -> List[str]:
        """Parses the SubgoalPlanner LLM's JSON output into a list of subgoals."""
        try:
            data = json.loads(llm_output)
            subgoal_list = data.get("subgoals", [])
            if not isinstance(subgoal_list, list):
                logger.error(f"SubgoalPlanner JSON output is malformed: 'subgoals' is not a list.\nOutput was:\n{llm_output}")
                return []

            logger.debug(f"SubgoalPlanner generated {len(subgoal_list)} subgoals.")
            return subgoal_list
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from SubgoalPlanner output: {e}\nOutput was:\n{llm_output}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred while parsing SubgoalPlanner output: {e}\nOutput was:\n{llm_output}")
            return []

    def _construct_zip_act_prompt(self, current_observation: str) -> List[Dict[str, str]]:
        """Constructs the prompt for the ZipAct LLM, including few-shot examples."""
        system_message = {
            "role": "system",
            "content": """You are a smart and efficient agent. Your goal is to complete the task described in the subgoals. Based on the state, subgoals, and observation, decide your next action.

You MUST respond with a JSON object of the format: {"thought": "your brief reasoning for the action", "action": "the single, specific action to execute next"}"""
        }
        
        messages = [system_message]
        for example in self.zip_act_examples:
            state = "\n".join(f"- {s}" for s in example["state_list"]) if example["state_list"] else "None"
            subgoals = "\n".join(f"- {g}" for g in example["subgoal_list"]) if example["subgoal_list"] else "None"
            user_prompt = f"""Current State:
{state}

Subgoals:
{subgoals}

Latest Observation:
{example["observation"]} """
            messages.append({"role": "user", "content": user_prompt})
            
            # The expected output should already be a JSON string in the ICL file, but if not, this would be the place to format it.
            # Assuming expected_output is a dict: e.g., {"thought": "...", "action": "..."}
            # assistant_response = json.dumps(example["expected_output"])
            # However, the current ICL format seems to have the JSON string directly.
            messages.append({"role": "assistant", "content": example["expected_output"]})

        formatted_state = "\n".join(f"- {s}" for s in self.state_list) if self.state_list else "None"
        formatted_subgoals = "\n".join(f"- {g}" for g in self.subgoal_list) if self.subgoal_list else "None"
        final_user_prompt = f"""Current State:
{formatted_state}

Subgoals:
{formatted_subgoals}

Latest Observation:
{current_observation}"""
        messages.append({"role": "user", "content": final_user_prompt})

        return messages

    def _parse_zip_act_response(self, llm_output: str) -> Tuple[str, str]:
        """Parses the ZipAct LLM's JSON output into think and action."""
        try:
            data = json.loads(llm_output)
            think = data.get("thought", "")
            action = data.get("action", "")

            if not action:
                logger.warning("ZipAct JSON output is missing 'action'. Using fallback.")
                return think, "look" # Fallback action

            return think, action
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from ZipAct output: {e}\nOutput was:\n{llm_output}")
            return "", "look" # Fallback action
        except Exception as e:
            logger.error(f"An unexpected error occurred while parsing ZipAct output: {e}\nOutput was:\n{llm_output}")
            return "", "look" # Fallback action

    def __call__(self, messages: List[Dict[str, str]], turn_num: int) -> Dict[str, Any]:
        """Executes the ZipAct agent's logic for a single turn with structured logging."""
        current_observation = messages[-1]['content']
        
        # Usage tracking for the turn
        planner_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        updater_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        zip_act_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if self.is_first_turn:
            goal_description_raw = current_observation.split("Your task is to:")[-1].strip()
            goal_description = goal_description_raw.split("This workflow may be helpful")[0].strip()

            try:
                # --- SubgoalPlanner Phase ---
                prompt_messages = self._construct_subgoal_planner_prompt(goal_description)
                prompt_text_for_log = "\n".join([f"<{msg['role']}>\n{msg['content']}" for msg in prompt_messages])
                
                logger.info(f"\n{Fore.GREEN}----------------- [SUBGOAL PLANNER] ------------------{Fore.RESET}")
                logger.info(f"{Fore.YELLOW}[INPUT]{Fore.RESET}\n{prompt_text_for_log}")

                subgoal_plan_str, usage = self._call_llm_json(
                    self.state_updater_client, self.state_updater_model_name, prompt_messages
                )
                
                logger.info(f"{Fore.YELLOW}[OUTPUT]{Fore.RESET}\n{subgoal_plan_str}")
                
                self.subgoal_list = self._parse_subgoal_planner_response(subgoal_plan_str)
                if not self.subgoal_list:
                    self.subgoal_list.append(f"Complete the task: {goal_description}")
                
                planner_usage = usage.__dict__
            except (openai.APIError, openai.Timeout, openai.RateLimitError) as e:
                logger.error(f"SubgoalPlanner failed due to an API/network issue: {e}. This may be retried by 'backoff'.")
                raise e # Re-raise to allow backoff to handle it, or to terminate the turn.
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"FATAL: SubgoalPlanner failed due to a code or data error: {e}. Check agent code and ICL examples.")
                raise e # Re-raise to terminate the entire run.
            
            self.state_list.append(f"Task: {goal_description}")
        else:
            try:
                # --- StateUpdater Phase ---
                last_action = messages[-2]['content']
                prompt_messages = self._construct_state_updater_prompt(last_action, current_observation)
                prompt_text_for_log = "\n".join([f"<{msg['role']}>\n{msg['content']}" for msg in prompt_messages])

                logger.info(f"\n{Fore.MAGENTA}------------------ [STATE UPDATER] -------------------{Fore.RESET}")
                logger.info(f"{Fore.YELLOW}[INPUT]{Fore.RESET}\n{prompt_text_for_log}")

                new_memory_str, usage = self._call_llm_json(self.state_updater_client, self.state_updater_model_name, prompt_messages)
                
                logger.info(f"{Fore.YELLOW}[OUTPUT]{Fore.RESET}\n{new_memory_str}")

                self.state_list, self.subgoal_list = self._parse_new_memory(new_memory_str)
                updater_usage = usage.__dict__
            except (openai.APIError, openai.Timeout, openai.RateLimitError) as e:
                logger.error(f"StateUpdater failed due to an API/network issue: {e}. This may be retried by 'backoff'.")
                raise e
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"FATAL: StateUpdater failed due to a code or data error: {e}. Check agent code and ICL examples.")
                raise e

        # --- ZipAct (Decision) Phase ---
        action = "Action: look"  # Fallback action
        try:
            prompt_messages = self._construct_zip_act_prompt(current_observation)
            prompt_text_for_log = "\n".join([f"<{msg['role']}>\n{msg['content']}" for msg in prompt_messages])
            
            logger.info(f"\n{Fore.CYAN}---------------------- [ZIP ACT] -----------------------{Fore.RESET}")
            logger.info(f"{Fore.YELLOW}[INPUT]{Fore.RESET}\n{prompt_text_for_log}")

            zip_act_response, usage = self._call_llm_json(self.zip_act_client, self.zip_act_model_name, prompt_messages)

            logger.info(f"{Fore.YELLOW}[OUTPUT]{Fore.RESET}\n{zip_act_response}")
            
            think, parsed_action = self._parse_zip_act_response(zip_act_response)
            action = f"Action: {parsed_action}"
            zip_act_usage = usage.__dict__
        except (openai.APIError, openai.Timeout, openai.RateLimitError) as e:
            logger.error(f"ZipAct failed due to an API/network issue: {e}. This may be retried by 'backoff'.")
            raise e
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"FATAL: ZipAct failed due to a code or data error: {e}. Check agent code and ICL examples.")
            raise e

        self.is_first_turn = False
        
        # The final action is logged in main.py after the full turn summary
        total_usage = {
            "planner": planner_usage,
            "updater": updater_usage,
            "zip_act": zip_act_usage
        }
        return {"content": action, "usage": total_usage}