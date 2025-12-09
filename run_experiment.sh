# Task configuration
AGENT_TYPE="zip_act" # agent (ReAct), zip_act
TASK_TYPE="alfworld" # alfworld, sciworld
TEST_MODEL="gpt-4o-mini-ca" 
METAPLAN_TYPE="mpo" # none, sft, rft, mpo
INCORPORATION_TYPE="query" # query, observation, thought
MODEL_PATH="/mnt/weimin/models" # the path to the model
SPLIT="test" # test, dev

start_port=8000
start_device=0

# Function to kill all background processes
cleanup() {
    echo "Caught Ctrl-C, killing all background processes..."
    pkill -P $$
    exit 1
}

# Set up the trap to catch Ctrl-C (SIGINT)
trap cleanup SIGINT

# Create log directory
if [ ! -d "logs" ]; then
    mkdir logs
fi

# Serve the model
if [[ $TEST_MODEL != *"gpt"* ]]; then
    CUDA_VISIBLE_DEVICES=$start_device vllm serve $MODEL_PATH/$TEST_MODEL --port $start_port > logs/${TASK_TYPE}_${TEST_MODEL}_${METAPLAN_TYPE}_${INCORPORATION_TYPE}.log 2>&1 &
    sleep 60
    api_base="http://localhost:$start_port/v1"
    api_key="EMPTY"
    echo "Start the model $TEST_MODEL on localhost:$start_port"
else
    api_base="https://api.chatanywhere.tech/v1" # TODO: replace with your own api base
    api_key="sk-iqbz8eE88kzff6lzhvzM49k34JF7pS0jKeDHqFx16FncY4An" # TODO: replace with your own api key
fi


# Run the experiment
echo "Start running experiments: $TASK_TYPE $METAPLAN_TYPE $INCORPORATION_TYPE"
echo "Evaluation model: $TEST_MODEL"
if [[ $METAPLAN_TYPE == "none" ]]; then
    python main.py --exp_config $TASK_TYPE --agent_config $AGENT_TYPE --model_name $TEST_MODEL --split $SPLIT --metaplan_type $METAPLAN_TYPE  --incorporation_type $INCORPORATION_TYPE --api_base $api_base --api_key $api_key --debug
else
    python main.py --exp_config $TASK_TYPE --agent_config $AGENT_TYPE --model_name $TEST_MODEL --split $SPLIT --metaplan_type $METAPLAN_TYPE --metaplan_path data/$TASK_TYPE/metaplan/metaplan_${SPLIT}_${METAPLAN_TYPE}.jsonl --incorporation_type $INCORPORATION_TYPE --api_base $api_base --api_key $api_key --debug
fi

# Kill all background processes
pkill -P $$
exit 0