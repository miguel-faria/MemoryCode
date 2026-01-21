#!/bin/bash

model="/mnt/scratch-hades/miguelfaria/models/Tower-Plus-72B"
data_dir="./data"
cache_dir="./cache"
model_name="${model##*/}"
n_gpus=2
connection_mode="local"
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
touch "$instruction_session_dir/completed_${model_name}_sessions.txt"
for dialogue_id in {1..360}; do
    python -m fire code/generate_model_output.py generate_model_output_session \
        --dialogue_file "$data_dir/memory_code/dataset/dialogue_${dialogue_id}.json" \
        --model "$model_name" \
        --instruction_output_path "$instruction_dir/${model_name}.json" \
        --output_dir "$instruction_session_dir" \
        --connection_mode "$connection_mode" \
        --n_gpus "$n_gpus" \
        --cache_path "$cache_dir"
done

# History
touch "$instruction_history_dir/completed_${model_name}_histories.txt"
for dialogue_id in {1..360}; do
    python -m fire code/generate_model_output.py generate_model_output_history \
        --dialogue_file "$data_dir/memory_code/dataset/dialogue_${dialogue_id}.json" \
        --model "$model_name" \
        --instruction_session_path "$instruction_session_dir/${model_name}/output_${dialogue_id}.json" \
        --output_dir "$instruction_history_dir" \
        --connection_mode "$connection_mode" \
        --n_gpus "$n_gpus" \
        --cache_path "$cache_dir"
done
