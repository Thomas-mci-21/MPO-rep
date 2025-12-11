import argparse
import json
import logging
import os
import pathlib
from typing import Any, Dict

from colorama import Fore
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import agents as agents
import envs as envs
import tasks as tasks
from utils.datatypes import State

logger = logging.getLogger("agent_eval")
args = None

def interactive_loop(
    task: tasks.Task,
    agent: agents.BaseAgent,
    env_config: Dict[str, Any],
) -> State:
    total_prompt_tokens = 0
    total_completion_tokens = 0

    logger.info(f"Loading environment: {env_config['env_class']}")
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
    env.args = args
    # reset the environment and set the prompt
    observation, state = env.reset()

    # The initial goal/task description
    logger.info(f"\n{Fore.YELLOW}{observation}{Fore.RESET}")

    while not state.finished:
        logger.info(f"\n{Fore.YELLOW}========================= TURN {state.steps} =========================")
        
        # Log the current observation from the environment
        if state.steps > 1:
             logger.info(f"{Fore.BLUE}[OBSERVATION]{Fore.RESET}\n{observation}")

        if args.incorporation_type == "thought"  and task.workflow:
            agent.set_workflow(f"This workflow maybe helpful to complete the task:\n{task.workflow}\n")
        try:
            # Agent performs its logic for the turn
            if args.incorporation_type == "thought" and task.workflow:
                response = agent.call_with_workflow(state.history) # Note: This path might need similar updates
            else:
                response = agent(state.history, state.steps)

            llm_output = response["content"]
            usage = response.get("usage", {})
            
            # Log the final action chosen by the agent for this turn
            logger.info(f"\n{Fore.GREEN}[ACTION] ==> {llm_output.replace('Action: ', '')}{Fore.RESET}")

            # Log the consolidated token usage for the turn
            if usage:
                p_usage = usage.get('planner', {})
                u_usage = usage.get('updater', {})
                z_usage = usage.get('zip_act', {})
                
                total_prompt = p_usage.get('prompt_tokens', 0) + u_usage.get('prompt_tokens', 0) + z_usage.get('prompt_tokens', 0)
                total_completion = p_usage.get('completion_tokens', 0) + u_usage.get('completion_tokens', 0) + z_usage.get('completion_tokens', 0)
                
                total_prompt_tokens += total_prompt
                total_completion_tokens += total_completion

                token_log = (
                    f"[TOKENS] Planner: {p_usage.get('total_tokens', 0)}, "
                    f"StateUpdater: {u_usage.get('total_tokens', 0)}, "
                    f"ZipAct: {z_usage.get('total_tokens', 0)} | "
                    f"TOTAL: {total_prompt + total_completion}"
                )
                logger.info(f"{Fore.CYAN}{token_log}{Fore.RESET}")

        except Exception as e:
            logger.info(f"Agent failed with error: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = "exceeding maximum input length"
            break

        # Log the end of the turn
        logger.info(f"{Fore.YELLOW}========================================================={Fore.RESET}")

        # Environment steps forward with the agent's action
        observation, state = env.step(llm_output)

        if state.finished:
            break

    if state.reward is not None:
        logger.info(
            f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}"
        )
    else:
        logger.info(
            f"Task finished in {state.steps} steps. Success: {state.success}"
        )

    logger.info(f"\n{Fore.CYAN}--- Token Usage Summary ---")
    logger.info(f"Total Prompt Tokens:     {total_prompt_tokens}")
    logger.info(f"Total Completion Tokens: {total_completion_tokens}")
    logger.info(f"Grand Total:             {total_prompt_tokens + total_completion_tokens}{Fore.RESET}")

    return state


def main(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)
    
    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name
    if args.api_base is not None:
        agent_config['config']['api_base'] = args.api_base
    if args.api_key is not None:
        agent_config['config']['api_key'] = args.api_key

    exp_name = args.exp_name or args.exp_config

    if args.output_dir is not None:
        output_path = args.output_dir
    else:
        if args.split == "test":
            output_path = os.path.join("unseen", agent_config['config']['model_name'].split('/')[-1], exp_name, args.metaplan_type)
        elif args.split == "dev":
            output_path = os.path.join("seen", agent_config['config']['model_name'].split('/')[-1], exp_name, args.metaplan_type)
        # ...
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
        print(f"DEBUG: The absolute output path is: {os.path.abspath(output_path)}")
    
        file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode='a')
        logging.basicConfig(
            format="%(message)s",
            handlers=[logging.StreamHandler(), file_handler],
        )

    env_config = exp_config["env_config"]
    
    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")


    if env_config['env_class'] == 'SciWorldEnv':
        from scienceworld import ScienceWorldEnv
        from utils.replace_sciworld_score import sciworld_monkey_patch
        sciworld_monkey_patch()
        env_config['env'] = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_config['env_jar_path']), envStepLimit=200)

    # initialize all the tasks    
    task_config: Dict[str, Any] = exp_config["task"]
    if args.metaplan_path is not None:
        task_config['workflow_path'] = args.metaplan_path
    
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(
        path=task_config.get("filepath", ""),
        workflow_path=task_config.get("workflow_path", None),
        split=args.split,
        part_num=args.part_num,
        part_idx=args.part_idx,
    )

    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    state_list = []
    
    done_task_id = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith('json'):
                continue
            state = State.load_json(json.load(open(os.path.join(output_path, file))))
            state_list.append(state)
            done_task_id.append(int(file.split('.')[0]))
        logger.info(f"Existing output file found. {len(done_task_id)} tasks done.")


    if len(done_task_id) == n_tasks:
        reward_list = []
        success_list = []
        for state in state_list:
            if state.reward is not None:
                reward_list.append(state.reward)
            success_list.append(state.success)

        if len(reward_list) != 0:
            logger.info(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
        logger.info(f"Success rate: {sum(success_list)/len(success_list):.4f}")
        logger.info("All tasks done. Exiting.")
        return

    # run the loop for all tasks
    logging.info(f"Running interactive loop for {n_tasks} tasks.")
    n_todo_tasks = n_tasks - len(done_task_id)  # only run the remaining tasks

    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)
        for i, task in enumerate(all_tasks):
            # Only test 10 tasks in debug mode
            if args.debug and i == 1:
                break

            # skip done tasks
            if task.task_id in done_task_id:
                continue

            state = interactive_loop(
                task, agent, env_config
            )

            state_list.append(state)
            json.dump(state.to_dict(), open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)

            pbar.update(1)
        pbar.close()
        
        logger.info("All tasks done.")
        logger.info(f"Output saved to {output_path}")

        # calculate metrics
        reward_list = []
        success_list = []
        
        # Check if this was a ScienceWorld experiment
        is_sciworld = exp_config.get("env_config", {}).get("env_class") == "SciWorldEnv"

        for state in state_list:
            if state.reward is not None:
                reward_list.append(state.reward)
            
            if is_sciworld:
                # For ScienceWorld, apply the fix from the ReflAct paper.
                # A task is successful only if the final progress reward is >= 0.7.
                is_successful = state.reward is not None and state.reward >= 0.7
                success_list.append(is_successful)
            else:
                # For other environments like ALFWorld, use the default success flag.
                success_list.append(state.success)

        # AR (Average Reward) - The average of all progress scores
        if len(reward_list) != 0:
            logger.info(f"Average reward (AR): {sum(reward_list)/len(reward_list):.4f}")
        
        # SR (Success Rate) - The percentage of tasks meeting the success criteria
        if len(success_list) != 0:
            logger.info(f"Success rate (SR): {sum(success_list)/len(success_list):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./configs/task",
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="math",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Config of experiment.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=-1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default="./configs/model",
        help="Config path of model.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="checkpoint-796-llama",
        help="Config of model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="Model name. It will override the 'model_name' in agent_config"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--incorporation_type",
        type=str,
        choices=["query", "observation", "thought"],
        default="query",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        required=False,
        help="Agent base url. It will override the 'api_base' in agent_config",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=False,
        help="Agent api key. It will override the 'api_key' in agent_config",
    )
    parser.add_argument(
        "--metaplan_path",
        type=str,
        required=False,
        help="Metaplan path.",
    )
    parser.add_argument(
        "--metaplan_type",
        type=str,
        choices=["none", "sft", "rft", "mpo"],
        required=True,
        help="Metaplan type.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Directory to save the output.",
    )
    
    
    args = parser.parse_args()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    main(args)
