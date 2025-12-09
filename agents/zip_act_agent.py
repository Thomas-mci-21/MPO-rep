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
5. Output the complete, updated lists, even if no changes were made.

OUTPUT FORMAT:
<New State>
- [State item 1]
...
</New State>
<New Subgoals>
- [Subgoal item 1]
...
</New Subgoals>"""
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

TASK: Based on the context, provide the updated memory."""
            messages.append({"role": "user", "content": user_prompt})

            new_state = "\n".join(f"- {s}" for s in example["expected_new_state_list"]) if example["expected_new_state_list"] else "None"
            new_subgoals = "\n".join(f"- {g}" for g in example["expected_new_subgoal_list"]) if example["expected_new_subgoal_list"] else "None"
            assistant_response = f"""<New State>
{new_state}
</New State>
<New Subgoals>
{new_subgoals}
</New Subgoals>"""
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

TASK: Based on the context, provide the updated memory."""
        messages.append({"role": "user", "content": final_user_prompt})
        
        return messages

    def _parse_new_memory(self, llm_output: str) -> Tuple[List[str], List[str]]:
        """Parses the StateUpdater LLM's output into state and subgoal lists."""
        try:
            state_match = re.search(r"<New State>(.*?)</New State>", llm_output, re.DOTALL)
            states_str = state_match.group(1).strip() if state_match else ""
            new_state_list = [line.strip().lstrip('- ') for line in states_str.split('\n') if line.strip() and line.strip().lower() != 'none']

            subgoal_match = re.search(r"<New Subgoals>(.*?)</New Subgoals>", llm_output, re.DOTALL)
            subgoals_str = subgoal_match.group(1).strip() if subgoal_match else ""
            new_subgoal_list = [line.strip().lstrip('- ') for line in subgoals_str.split('\n') if line.strip() and line.strip().lower() != 'none']
            
            logger.info(f"{Fore.MAGENTA}StateUpdater parsed memory. New state items: {len(new_state_list)}, New subgoal items: {len(new_subgoal_list)}{Fore.RESET}")
            return new_state_list, new_subgoal_list
        except Exception as e:
            logger.error(f"Failed to parse StateUpdater output: {e}\nOutput was:\n{llm_output}")
            return self.state_list, self.subgoal_list

    def _construct_zip_act_prompt(self, current_observation: str) -> List[Dict[str, str]]:
        """Constructs the prompt for the ZipAct LLM, including few-shot examples."""
        system_message = {
            "role": "system",
            "content": """You are a smart and efficient agent. Your goal is to complete the task described in the subgoals. Based on the state, subgoals, and observation, decide your next action. First, provide a brief thought process, then specify the exact action.

OUTPUT FORMAT:
THINK: [Your brief reasoning for the action.]
ACTION: [The single, specific action to execute next.]"""
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
        """Parses the ZipAct LLM's output into think and action."""
        try:
            think_match = re.search(r"THINK:(.*?)ACTION:", llm_output, re.DOTALL)
            action_match = re.search(r"ACTION:(.*)", llm_output, re.DOTALL)

            think = think_match.group(1).strip() if think_match else ""
            action = action_match.group(1).strip() if action_match else llm_output

            if not action:
                logger.warning("Could not parse ACTION from ZipAct output. Using entire output as action.")
                action = llm_output.strip()

            return think, action
        except Exception as e:
            logger.error(f"Failed to parse ZipAct output: {e}\nOutput was:\n{llm_output}")
            return "", llm_output.strip()

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Executes the ZipAct + StateUpdater loop."""
        logger.info(f"{Fore.YELLOW}--- Turn Start ---
{Fore.RESET}")
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        current_observation = messages[-1]['content']
        
        if self.is_first_turn:
            logger.info("First turn: Initializing memory from task description.")
            # Ensure the initial observation is clean and just the task goal
            goal_description_raw = current_observation.split("Your task is to:")[-1].strip()
            # Remove workflow part if it somehow got included (though env is supposed to prevent it now)
            goal_description = goal_description_raw.split("This workflow may be helpful")[0].strip()

            # Attempt to parse a more structured goal
            goal_match = re.search(r"heat some (\w+) and put it in (\w+)\", goal_description)
            if goal_match:
                item_to_heat = goal_match.group(1)
                location_to_put = goal_match.group(2)
                self.subgoal_list.append(f"Find and take {item_to_heat}.")
                self.subgoal_list.append(f"Heat {item_to_heat}.")
                self.subgoal_list.append(f"Put {item_to_heat} in {location_to_put}.")
                self.state_list.append(f"Task: {goal_description}")
            else:
                self.subgoal_list.append(f"Complete the task: {goal_description}")
            self.state_list.append("The initial state is described by the observation and subgoals.")
        else:
            logger.info(f"{Fore.MAGENTA}--- Running StateUpdater Phase ---
{Fore.RESET}")
            try:
                last_action = messages[-2]['content']
                logger.debug(f"StateUpdater INPUT - Last Action: {last_action}")
                logger.debug(f"StateUpdater INPUT - New Observation: {current_observation}")

                state_updater_prompt = self._construct_state_updater_prompt(last_action, current_observation)
                
                new_memory_str, usage = self._call_llm(self.state_updater_client, self.state_updater_model_name, state_updater_prompt)
                logger.debug(f"StateUpdater RAW OUTPUT:\n{new_memory_str}")
                
                self.state_list, self.subgoal_list = self._parse_new_memory(new_memory_str)
                
                total_usage["prompt_tokens"] += usage.prompt_tokens
                total_usage["completion_tokens"] += usage.completion_tokens
                total_usage["total_tokens"] += usage.total_tokens
                logger.info(f"{Fore.MAGENTA}StateUpdater Tokens: {usage.prompt_tokens} (P) + {usage.completion_tokens} (C) = {usage.total_tokens} (T){Fore.RESET}")

            except (IndexError, KeyError) as e:
                logger.error(f"Could not extract context for StateUpdater (is history correct?): {e}")
            except Exception as e:
                logger.error(f"StateUpdater LLM call failed. Details logged in _call_llm.")

        logger.info(f"{Fore.CYAN}--- Running ZipAct (Decision) Phase ---
{Fore.RESET}")
        logger.debug(f"ZipAct INPUT - State: {self.state_list}")
        logger.debug(f"ZipAct INPUT - Subgoals: {self.subgoal_list}")
        logger.debug(f"ZipAct INPUT - Observation: {current_observation}")

        zip_act_prompt = self._construct_zip_act_prompt(current_observation)
        
        action = "Action: look" # FIX: Correctly formatted fallback action
        try:
            zip_act_response, usage = self._call_llm(self.zip_act_client, self.zip_act_model_name, zip_act_prompt)
            logger.debug(f"ZipAct RAW OUTPUT:\n{zip_act_response}")
            
            think, parsed_action = self._parse_zip_act_response(zip_act_response)
            action = f"Action: {parsed_action}" # Ensure the action is always formatted correctly
            logger.info(f"{Fore.GREEN}ZipAct THINK: {think}{Fore.RESET}")
            
            total_usage["prompt_tokens"] += usage.prompt_tokens
            total_usage["completion_tokens"] += usage.completion_tokens
            total_usage["total_tokens"] += usage.total_tokens
            logger.info(f"{Fore.CYAN}ZipAct Tokens: {usage.prompt_tokens} (P) + {usage.completion_tokens} (C) = {usage.total_tokens} (T){Fore.RESET}")

        except Exception as e:
            logger.error(f"ZipAct LLM call failed. Details logged in _call_llm.")

        self.is_first_turn = False
        
        logger.info(f"{Fore.BLUE}==> Final Action: {action}{Fore.RESET}")
        return {"content": action, "usage": total_usage}