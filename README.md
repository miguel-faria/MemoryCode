# MemoryCode  
<img align="middle" src="figures/dataset_creation.png" alt="dataset creation">
## Key terms
- A **conversation** is composed of multiple **sessions**. A session is composed of multiple turns.
- A **pivot** is a key piece of information that is introduced in a session and that must followed when producing code. It can be updated throughout the conversation. Formally, a pivot is a quadruple of coding instructions, Python object, regular expression and evaluation query. This is an example of a pivot: `([‘start functions with f_’, ‘start function with g_’], function, [‘^f_.*’, ‘^g_.*’], function that merges two lists)`. 
- A **filler** is a conversation topic not related to coding instructions. It can also be updated during the conversation.

## Dataset generation
Dataset generation can be divided into 3 stages: template generation, prompt generation, conversation evaluation. 

The `topics.json` file contains the list of all instructions, fillers, names and personas to sample from for conversation generation.

The `generate_template.py` script takes as input the `topics.json` file along with several parameters and produces a conversation template that is stored in `dataset`. Given a template, the `generate_prompt.py` script produces the corresponding prompt file in `prompts`. These prompts are then fed to an LLM using the `generate_conversation.py` script to produce the conversations.

Run the `scripts/generate_dataset.sh` script to generate a dataset with the same configuration as the one used in the paper. 

## Evaluation
Run the `scripts/generate_model_output.sh` script to generate the model outputs. The `evaluate_model_output.py` script takes as input the conversation directory, the model outputs directory and prints the scores. For example, to evaluate gpt-4o, run the following command:

`python code/evaluate_model_output.py --conversation_dir dataset --model_output_dir outputs/gpt-4o`

  

