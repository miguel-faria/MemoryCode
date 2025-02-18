set -e

export PYTHONPATH=$(pwd)/code:$PYTHONPATH
mkdir -p model_outputs/instruction
mkdir -p model_outputs/instruction_session
mkdir -p model_outputs/instruction_session_cumulative

model_name=gpt-4o
# Evaluation is split into 3 steps. The final output is stored in model_outputs/instruction_session_cumulative.

# Instructions
python -m fire code/generate_model_output.py generate_model_output_instruction --model_name "$model_name" \
    --topics_file topics.json \
    --output_dir model_outputs/instruction

# Sessions
for conversation_id in {1..360}; do
    python -m fire code/generate_model_output.py generate_model_output_session \
        --conversation_file "dataset/conversation_${conversation_id}.json" \
        --model_name "$model_name" \
        --instruction_output_path "model_outputs/instruction/${model_name}.json" \
        --output_dir model_outputs/instruction_session
done

# Cumulative
for conversation_id in {1..360}; do
    python -m fire code/generate_model_output.py generate_model_output_cumulative \
        --conversation_file "dataset/conversation_${conversation_id}.json" \
        --model_name "$model_name" \
        --instruction_session_path "model_outputs/instruction_session/${model_name}/output_${conversation_id}.json" \
        --output_dir model_outputs/instruction_session_cumulative
done
