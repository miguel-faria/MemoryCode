import json
import os
import cohere
import fire
import time

from generation_utils import cohere_model_generate, openai_model_generate
from tqdm import tqdm
from vllm import LLM, SamplingParams
from pathlib import Path
from openai import OpenAI


def generate_dialogue_history(prompt_dir, template_dir, connection_mode='cohere', model_name='meta-llama/Llama-3.3-70B-Instruct',
                              n_gpus=2, model_url='http://localhost:12000/v1', api_key='a1b2c3d4e5', cache_path=None):
    """Generates a dialogue given a prompt then populates the corresponding template dialogue file."""
    for prompt_file in Path(prompt_dir).iterdir():
        
        print('Generating dialogue for prompt file: ', prompt_file)
        completed_dialogues_file = os.path.join(template_dir, "completed_dialogues.txt")
        with open(completed_dialogues_file, "r") as f:
            completed_dialogues = f.read().splitlines()
        dialogue_id = prompt_file.name.split("_")[1].split(".")[0]
        if f"dialogue_{dialogue_id}" in completed_dialogues:
            print(f"Dialogue dialogue_{dialogue_id} already completed. Skipping...")
            continue
        
        dialogue_file = os.path.join(template_dir, f"dialogue_{dialogue_id}.json")
        assert os.path.exists(dialogue_file), f"Template file {dialogue_file} does not exist!"
        
        with open(prompt_file, "r") as f:
            prompt_data = json.load(f)
        with open(dialogue_file, "r") as f:
            template = json.load(f)
    
        ### Initialize the Cohere client
        if connection_mode == 'cohere':
            try:
                COHERE_API_KEY = os.environ["COHERE_API_KEY"]
            except KeyError:
                raise ValueError("You need to set the COHERE_API_KEY env variable!")
            client = cohere.Client(COHERE_API_KEY)
        
            ### Generate dialogue
            preamble = prompt_data["preamble"]
            i = 0
            print(f'Dialogue: dialogue_{dialogue_id}')
            for session, prompt in tqdm(zip(template["sessions"], prompt_data["prompts"])):
                session_text = cohere_model_generate(
                    prompt, client, preamble, model_name="command-r-plus-08-2024", temperature=0.9, p=0.9
                )
                session["text"] = session_text
                i += 1
                if i % 10 == 0:
                    time.sleep(10) # to avoid rate limiting
        
        elif connection_mode == 'gpt':
            client = OpenAI()
            preamble = prompt_data["preamble"]
            i = 0
            print(f'Dialogue: dialogue_{dialogue_id}')
            for session, prompt in tqdm(zip(template["sessions"], prompt_data["prompts"])):
                session_text = openai_model_generate(
                    client, prompt, preamble, model_name=model_name
                )
                session["text"] = session_text
                i += 1
                if i % 10 == 0:
                    time.sleep(10) # to avoid rate limiting
        
        elif connection_mode == 'vllm':
            client = OpenAI(base_url=model_url, api_key=api_key)
            preamble = prompt_data["preamble"]
            print(f'Dialogue: dialogue_{dialogue_id}')
            for session, prompt in tqdm(zip(template["sessions"], prompt_data["prompts"])):
                session_text = openai_model_generate(
                    client, prompt, preamble, model_name=model_name
                )
                session["text"] = session_text
        
        else:
            cache_dir = Path(cache_path) if cache_path is not None else Path(__file__).absolute().parent.parent.parent.parent / 'cache'
            model = LLM(model_name, gpu_memory_utilization=0.95, enforce_eager=True, download_dir=cache_dir, tensor_parallel_size=n_gpus, dtype='half', max_model_len=2048)
            gen_params = SamplingParams(
                    temperature=1.0,
                    top_k=5,
                    top_p=0.75,
                    max_tokens=1024,
            )
            
            ### Generate dialogue
            preamble = prompt_data["preamble"]
            print(f'Dialogue: dialogue_{dialogue_id}')
            for session, prompt in tqdm(zip(template["sessions"], prompt_data["prompts"])):
                messages = [{"role": "system", "content": preamble}, {"role": "user", "content": prompt}] if preamble != '' else [{"role": "user", "content": prompt}]
                session_text = model.chat(messages, gen_params)[0].outputs[0].text
                session["text"] = session_text
    
        ### save dialogue dataset
        with open(dialogue_file, "w") as f:
            json.dump(template, f, indent=2)
            
        with open(completed_dialogues_file, "a") as f:
            f.write(f"dialogue_{dialogue_id}\n")


if __name__ == "__main__":
    fire.Fire(generate_dialogue_history)
