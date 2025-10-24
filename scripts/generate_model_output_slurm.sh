#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=generate_memory_code
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
# #SBATCH --gres=gpu:quadro6000:1
# #SBATCH --gres=gpu:rtx2080:2
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=gpu-h200
#SBATCH --output="job-%x-%j.out"
#SBATCH --partition=h200

date;hostname;pwd

if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] || [ "$HOSTNAME" = "maia" ] || [ "$HOSTNAME" = "hades" ] ; then
  # cache_dir="/mnt/scratch-artemis/miguelfaria/llms/checkpoints"
  cache_dir="/mnt/scratch-hades/miguelfaria/models"
  data_dir="/mnt/data-artemis/miguelfaria/agentic_llm/"
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
model="/mnt/scratch-hades/miguelfaria/models/Tower-Plus-72B"
api_key="a1b2c3d4e5"
n_gpus=2
host="localhost"
port=12500
gpu_usage=0.75
model_url="http://$host:$port/v1"
connection_mode="local"

set -e

instruction_dir="$data_dir"/memory_code/model_outputs/instruction
instruction_session_dir="$data_dir"/memory_code/model_outputs/instruction_session
instruction_history_dir="$data_dir"/memory_code/model_outputs/instruction_session_history

export PYTHONPATH=$(pwd)/code:$PYTHONPATH
mkdir -p "$instruction_dir"
mkdir -p "$instruction_session_dir"
mkdir -p "$instruction_history_dir"

# Evaluation is split into 3 steps. The final output is stored in model_outputs/instruction_session_history.

# Instructions
python -m fire code/generate_model_output.py generate_model_output_instruction --model_name "$model" \
    --topics_file "$data_dir"/memory_code/topics.json \
    --output_dir "$instruction_dir" \
    --cache_path "$cache_dir" \
    --connection_mode "$connection_mode" \
    --n_gpus "$n_gpus" \

# Sessions
# for dialogue_id in {1..360}; do
#     python -m fire code/generate_model_output.py generate_model_output_session \
#         --dialogue_file "$data_dir/memory_code/dataset/dialogue_${dialogue_id}.json" \
#         --model_name "$model" \
#         --instruction_output_path "$instruction_dir/${model_name}.json" \
#         --output_dir "$instruction_session_dir"
# done

# History
# for dialogue_id in {1..360}; do
#     python -m fire code/generate_model_output.py generate_model_output_history \
#         --dialogue_file "$data_dir/memory_code/dataset/dialogue_${dialogue_id}.json" \
#         --model_name "$model" \
#         --instruction_session_path "$instruction_session_dir/${model_name}/output_${dialogue_id}.json" \
#         --output_dir "$instruction_history_dir"
# done
