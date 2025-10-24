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
#  cache_dir="/mnt/scratch-artemis/miguelfaria/llms/checkpoints"   # for medium and small models stored on artemis scratch
  cache_dir="/mnt/scratch-hades/miguelfaria/models"               # for big models stored on hades scratch
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

set -e

# Template generation
# echo "Generating templates ..."
# id=1
# for n in 1 2 3 4 5 10 15 20 30 40 50 100; do
#     for i in {1..30}; do
#         instruction_min=0.5
#         instruction_max=0.7
#         instruction_only_session_max=2
#         instruction_update_rate_max=0.7
#         if [[ n -eq 1 ]]; then
#             instruction_min=1
#         fi
#         if [[ n -lt 6 ]]; then
#             instruction_max=1
#             instruction_only_session_max=1
#         fi
#         if [[ n -ge 20 ]]; then
#             instruction_only_session_max=$((n / 5))
#         fi
#         if [[ n -ge 50 ]]; then
#             instruction_update_rate_max=0.5
#         fi
#         echo "$n $id $instruction_min $instruction_max $instruction_only_session_max"
#         python code/generate_template.py --template_id $id \
#             --n_session_min $n \
#             --n_session_max $n \
#             --proportion_of_session_with_instruction_min $instruction_min \
#             --proportion_of_session_with_instruction_max $instruction_max \
#             --n_instruction_only_session_min 0 \
#             --n_instruction_only_session_max $instruction_only_session_max \
#             --n_instruction_per_instruction_session_min 1 \
#             --n_instruction_per_instruction_session_max 2 \
#             --instruction_update_rate_min 0.3 \
#             --instruction_update_rate_max $instruction_update_rate_max \
#             --max_update_per_filler 3 \
#             --filler_instruction_rate_min 0.5 \
#             --filler_instruction_rate_max 0.7 \
#             --filler_instruction_update_rate_min 0.5 \
#             --filler_instruction_update_rate_max 0.8 \
#             --topics_file "$data_dir"/memory_code/topics.json \
#             --output_dir "$data_dir"/memory_code/dataset
#         id=$((id + 1))
#     done
# done

# Prompt generation
# echo "Generating prompts ..."
# for template_file in "$data_dir"/memory_code/dataset/*.json; do
#     python code/generate_prompt.py --template_file $template_file --output_dir "$data_dir"/memory_code/prompts
# done

# Dialogue generation
echo "Generating dialogues ..."
model="/mnt/scratch-hades/miguelfaria/models/Tower-Plus-72B"
api_key="a1b2c3d4e5"
n_gpus=2
host="localhost"
port=12500
gpu_usage=0.95
model_url="http://$host:$port/v1"
# echo "Serving model via vLLM"
# vllm serve "$model" --download-dir "$cache_dir" --dtype auto --api-key "$api_key" --gpu-memory-utilization "$gpu_usage" \
                    # --tensor-parallel-size "$n_gpus" --host "$host" --port "$port" --max-model-len 2048 --disable-log-requests --enforce-eager
# model_id=$!
# sleep 2.5m
echo "Starting dialogue generation script"
python code/generate_dialogue.py --prompt_dir "$data_dir"/memory_code/prompts --template_dir "$data_dir"/memory_code/dataset --connection_mode local --model_name "$model" --n_gpus "$n_gpus" --model_url "$model_url" --api_key "$api_key" --cache_path "$cache_dir"

# kill -9 "$model_id"
conda deactivate
date