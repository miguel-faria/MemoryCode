#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=generate_memory_code_instructions
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:2
#SBATCH --gres=gpu:rtx2080:0,gpu:quadro6000:2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=gpu-medium
#SBATCH --output="job-%x-%j.out"
# #SBATCH --partition=h200

date;hostname;pwd

if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] || [ "$HOSTNAME" = "maia" ] || [ "$HOSTNAME" = "hades" ] ; then
  cache_dir="/mnt/scratch-artemis/miguelfaria/llms/checkpoints"
  # cache_dir="/mnt/scratch-hades/miguelfaria/models"
  # cache_dir="/mnt/scratch-hades/shared/models"
  data_dir="/mnt/data-artemis/miguelfaria/agentic_llm"
else
  cache_dir="./cache"
  data_dir="./data"
fi

export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="/opt/cuda/bin:$PATH"
export HF_HOME="$cache_dir"
if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] || [ "$HOSTNAME" = "hades" ] ; then
  if [ -z "$CONDA_PREFIX_1" ] ; then
    conda_dir="$CONDA_PREFIX"
  else
    conda_dir="$CONDA_PREFIX_1"
  fi
else
  conda_dir="$CONDA_HOME"
fi

n_gpus=$(echo "${CUDA_VISIBLE_DEVICES:-""}" | tr ',' '\n' | wc -l)
source "$conda_dir"/bin/activate llm_env
# model=""
# model="/mnt/scratch-hades/shared/models/Llama-3.3-70B-Instruct"
# model="/mnt/scratch-hades/miguelfaria/models/Tower-Plus-72B"
model="Qwen/Qwen3-32B-AWQ"
connection_mode="vllm"
# n_gpus=4
api_key="a1b2c3d4e5"
host="localhost"
port=14500
gpu_usage=0.75
model_url="http://$host:$port/v1"

set -e

instruction_dir="$data_dir"/memory_code/model_outputs/instruction
instruction_session_dir="$data_dir"/memory_code/model_outputs/instruction_session
instruction_history_dir="$data_dir"/memory_code/model_outputs/instruction_session_history

export PYTHONPATH=$(pwd)/code:$PYTHONPATH
mkdir -p "$instruction_dir"
mkdir -p "$instruction_session_dir"
mkdir -p "$instruction_history_dir"
if [ "$HOSTNAME" = "maia" ] ; then
  vllm serve "$model" --download-dir "$cache_dir" \
                      --dtype float16 \
                      --api-key "$api_key" \
                      --gpu-memory-utilization "$gpu_usage" \
                      --tensor-parallel-size "$n_gpus" \
                      --host "$host" \
                      --port "$port" \
                      --max-model-len 2048 \
                      --enforce-eager &
                      # --reasoning-parser mistral \
                      # --tokenizer_mode mistral \
                      # --config_format mistral \
                      # --load_format mistral \
                      # --tool-call-parser mistral \
                      # --enable-auto-tool-choice \
                      # --limit-mm-per-prompt '{"image":10}' \
                      # --enable-reasoning \
                      # --reasoning-parser deepseek_r1 &
else
  vllm serve "$model" --download-dir "$cache_dir" \
                      --dtype auto \
                      --api-key "$api_key" \
                      --gpu-memory-utilization "$gpu_usage" \
                      --tensor-parallel-size "$n_gpus" \
                      --host "$host" \
                      --port "$port" \
                      --max-model-len 2048 \
                      --enforce-eager &
                      # --reasoning-parser mistral \
                      # --tokenizer_mode mistral \
                      # --config_format mistral \
                      # --load_format mistral \
                      # --tool-call-parser mistral \
                      # --enable-auto-tool-choice \
                      # --limit-mm-per-prompt '{"image":10}' \
                      # --enable-reasoning \
                      # --reasoning-parser deepseek_r1 &
fi
model_id=$!
sleep 10m                            

# Instructions
python -m fire code/generate_model_output.py generate_model_output_instruction --model "$model" \
    --topics_file "$data_dir"/memory_code/topics.json \
    --output_dir "$instruction_dir" \
    --cache_path "$cache_dir" \
    --connection_mode "$connection_mode" \
    --n_gpus "$n_gpus" \
    --model_url "$model_url" \
    --api_key "$api_key" \

kill -9 "$model_id"
conda deactivate
date