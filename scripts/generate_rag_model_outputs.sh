#!/bin/bash

model="/mnt/scratch-hades/miguelfaria/models/Tower-Plus-72B"
data_dir="./data"
cache_dir="./cache"
model_name="${model##*/}"
retriever_name="jinaai/jina-reranker-v3"
retrieval_mode="local"
connection_mode="local"
n_gpus=2
# api_key="a1b2c3d4e5"
# host="localhost"
# port=12500
# gpu_usage=0.75
# model_url="http://$host:$port/v1"

set -e

instruction_dir="$data_dir"/memory_code/model_outputs/instruction
instruction_session_dir="$data_dir"/memory_code/model_outputs/instruction_session
instruction_history_dir="$data_dir"/memory_code/model_outputs/instruction_session_history

export PYTHONPATH=$(pwd)/code:$PYTHONPATH
mkdir -p "$instruction_dir"
mkdir -p "$instruction_session_dir"
mkdir -p "$instruction_history_dir"

# Sessions
touch "$instruction_session_dir/completed_${model_name}_rag.txt"
for dialogue_id in {1..360}; do
    python -m fire code/generate_model_output.py generate_model_output_rag \
          --dialogue_file "$data_dir/memory_code/dataset/dialogue_${dialogue_id}.json" \
          --model "$model" \
          --output_dir "$instruction_session_dir" \
          --connection_mode "$connection_mode" \
          --n_gpus "$n_gpus" \
          --cache_path "$cache_dir" \
          --retrieval_mode "$retrieval_mode" \
          --retriever_name "$retriever_name"
done
