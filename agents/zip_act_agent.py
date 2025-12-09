import logging
import os
import re
from typing import List, Dict, Any, Tuple

import backoff
import openai
from openai import OpenAI

from .base import BaseAgent

logger = logging.getLogger("agent_eval")

class ZipActAgent(BaseAgent):
    """
    An agent that uses a two-phase approach:
    1. ZipAct (Decision): A fast-thinking LLM call to decide the next action.
    2. StateUpdater (Memory Update): A reflective LLM call to update the state and subgoals.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # --- LLM Client Initialization ---
        # Initialize client for the ZipAct (decision) phase
        zip_act_config = config.get("zip_act_llm_config", {})
        self.zip_act_client = OpenAI(
            base_url=zip_act_config.get("api_base", config.get("api_base")),
            api_key=zip_act_config.get("api_key", config.get("api_key", os.environ.get("OPENAI_API_KEY"))),
        )
        self.zip_act_model_name = zip_act_config.get("model_name", config.get("model_name"))

        # Initialize client for the StateUpdater (memory) phase
        state_updater_config = config.get("state_updater_llm_config", {})
        self.state_updater_client = OpenAI(
            base_url=state_updater_config.get("api_base", config.get("api_base")),
            api_key=state_updater_config.get("api_key", config.get("api_key", os.environ.get("OPENAI_API_KEY"))),
        )
        self.state_updater_model_name = state_updater_config.get("model_name", config.get("model_name"))

        # --- Memory Initialization ---
        self.state_list: List[str] = []
        self.subgoal_list: List[str] = []
        self.is_first_turn = True

    @backoff.on_exception(
        backoff.fibo,
        (openai.APIError, openai.Timeout, openai.RateLimitError, openai.APIConnectionError),
    )
    def _call_llm(self, client: OpenAI, model_name: str, messages: List[Dict[str, str]]) -> Tuple[str, Any]:
        """A generic, decorated method to call an LLM."""
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
        """Constructs the prompt for the StateUpdater LLM."""
        
        # Format current memory for the prompt
        formatted_state = "\n".join(f"- {s}" for s in self.state_list) if self.state_list else "None"
        formatted_subgoals = "\n".join(f"- {g}" for g in self.subgoal_list) if self.subgoal_list else "None"

        prompt = f"""You are a meticulous memory assistant for an agent. Your task is to update the agent's memory based on the last action and the new observation.

RULES:
1. Update the 'Current State' and 'Subgoals' based *only* on the evidence from 'Action Taken' and 'New Observation'.
2. Edits should be minimal. Do not remove information unless the observation clearly invalidates it.
3. Do not invent new entities or facts.
4. If a subgoal is completed, remove it. A subgoal is completed only if the observation provides clear evidence.
5. Output the complete, updated lists for 'New State' and 'New Subgoals', even if no changes were made.

PREVIOUS MEMORY:
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

TASK:
Based on the context, provide the updated memory.

OUTPUT FORMAT:
<New State>
- [Updated state item 1]
- [Updated state item 2]
...
</New State>
<New Subgoals>
- [Updated subgoal item 1]
- [Updated subgoal item 2]
...
</New Subgoals>
"""
        return [{"role": "user", "content": prompt}]

    def _parse_new_memory(self, llm_output: str) -> Tuple[List[str], List[str]]:
        """Parses the StateUpdater LLM's output into state and subgoal lists."""
        new_state_list = []
        new_subgoal_list = []

        try:
            state_match = re.search(r"<New State>(.*?)</New State>", llm_output, re.DOTALL)
            if state_match:
                states_str = state_match.group(1).strip()
                if states_str.lower() != 'none':
                    new_state_list = [line.strip().lstrip('- ') for line in states_str.split('\n') if line.strip()]

            subgoal_match = re.search(r"<New Subgoals>(.*?)</New Subgoals>", llm_output, re.DOTALL)
            if subgoal_match:
                subgoals_str = subgoal_match.group(1).strip()
                if subgoals_str.lower() != 'none':
                    new_subgoal_list = [line.strip().lstrip('- ') for line in subgoals_str.split('\n') if line.strip()]
            
            logger.info(f"StateUpdater updated memory. New state items: {len(new_state_list)}, New subgoal items: {len(new_subgoal_list)}")
            return new_state_list, new_subgoal_list
        except Exception as e:
            logger.error(f"Failed to parse StateUpdater output: {e}\nOutput was:\n{llm_output}")
            # On failure, return the old memory to avoid corruption
            return self.state_list, self.subgoal_list

    def _construct_zip_act_prompt(self, current_observation: str) -> List[Dict[str, str]]:
        """Constructs the prompt for the ZipAct LLM."""
        
        formatted_state = "\n".join(f"- {s}" for s in self.state_list) if self.state_list else "None"
        formatted_subgoals = "\n".join(f"- {g}" for g in self.subgoal_list) if self.subgoal_list else "None"
        
        prompt = f"""You are a smart and efficient agent. Your goal is to complete the task described in the subgoals.

Current State:
{formatted_state}

Subgoals:
{formatted_subgoals}

Latest Observation:
{current_observation}

Based on the state, subgoals, and observation, decide your next action. First, provide a brief thought process (a few sentences), then specify the exact action.

OUTPUT FORMAT:
THINK: [Your brief reasoning for the action.]
ACTION: [The single, specific action to execute next.]
"""
        return [{"role": "user", "content": prompt}]

    def _parse_zip_act_response(self, llm_output: str) -> Tuple[str, str]:
        """Parses the ZipAct LLM's output into think and action."""
        try:
            think_match = re.search(r"THINK:(.*?)ACTION:", llm_output, re.DOTALL)
            action_match = re.search(r"ACTION:(.*)", llm_output, re.DOTALL)

            think = think_match.group(1).strip() if think_match else ""
            action = action_match.group(1).strip() if action_match else llm_output # Fallback

            if not action:
                # If parsing fails, assume the whole output is the action
                logger.warning("Could not parse ACTION from ZipAct output. Using entire output as action.")
                action = llm_output.strip()

            return think, action
        except Exception as e:
            logger.error(f"Failed to parse ZipAct output: {e}\nOutput was:\n{llm_output}")
            return "", llm_output.strip() # Return empty thought and full output as action

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Executes the ZipAct + StateUpdater loop.
        """
        # --- Phase A: Memory Update (StateUpdater) ---
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # StateUpdater runs from the second turn onwards
        if not self.is_first_turn:
            # Extract context from message history
            current_observation = messages[-1]['content']
            # The previous action was our own last output, which is the second to last message.
            last_action = messages[-2]['content']

            state_updater_prompt = self._construct_state_updater_prompt(last_action, current_observation)
            
            try:
                new_memory_str, usage = self._call_llm(self.state_updater_client, self.state_updater_model_name, state_updater_prompt)
                self.state_list, self.subgoal_list = self._parse_new_memory(new_memory_str)
                total_usage["prompt_tokens"] += usage.prompt_tokens
                total_usage["completion_tokens"] += usage.completion_tokens
                total_usage["total_tokens"] += usage.total_tokens
            except Exception as e:
                logger.error(f"StateUpdater LLM call failed: {e}")

        # --- Phase B: Decision (ZipAct) ---
        current_observation = messages[-1]['content']
        # On the first turn, the observation is the initial task description.
        # Initialize subgoals and state with this description.
        if self.is_first_turn:
            self.subgoal_list.append(f"Complete the task: {current_observation.split('GOAL: ')[-1].strip()}")
            self.state_list.append("The initial state is described by the observation.")

        zip_act_prompt = self._construct_zip_act_prompt(current_observation)
        
        action = ""
        try:
            zip_act_response, usage = self._call_llm(self.zip_act_client, self.zip_act_model_name, zip_act_prompt)
            think, action = self._parse_zip_act_response(zip_act_response)
            logger.info(f"ZipAct THINK: {think}")
            
            total_usage["prompt_tokens"] += usage.prompt_tokens
            total_usage["completion_tokens"] += usage.completion_tokens
            total_usage["total_tokens"] += usage.total_tokens
        except Exception as e:
            logger.error(f"ZipAct LLM call failed: {e}")
            action = "look" # Fallback action on failure

        # --- Phase C: Finalize ---
        self.is_first_turn = False
        
        return {
            "content": action,
            "usage": total_usage 
        }
