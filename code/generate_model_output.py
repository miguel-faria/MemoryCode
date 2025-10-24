import json
import os

import json
import os
import cohere
import fire
import time

from generation_utils import get_model_generate_function, perform_retrieval, cohere_model_generate, openai_model_generate
from tqdm import tqdm
from vllm import LLM, SamplingParams
from pathlib import Path
from openai import OpenAI


def generate_model_output_instruction(model_name, topics_file, output_dir, connection_mode='cohere', n_gpus=2,
                                      model_url='http://localhost:12000/v1', api_key='a1b2c3d4e5', cache_path=None):
    
    with open(topics_file, "r") as f:
        topics = json.load(f)
    instructions = topics["instructions"]
    
    preamble = f"""
    ## Style Guide
    Do not acknowledge. Only generate Python code and nothing else before or after. Do not explain the code. Do not ask for more information but directly give the answer.
    """
    prompt_template = f"Write a [eval_query]. Do not provide example usage. Follow this coding style guide when writing the code: [instruction_topic]."
    
    if connection_mode not in ['cohere', 'openai', 'vllm']:
        cache_dir = Path(cache_path) if cache_path is not None else Path(__file__).absolute().parent.parent.parent.parent / 'cache'
        model = LLM(model_name, gpu_memory_utilization=0.95, enforce_eager=True, download_dir=str(cache_dir), tensor_parallel_size=n_gpus, dtype='auto', max_model_len=2048)
        gen_params = SamplingParams(
                temperature=1.0,
                top_k=5,
                top_p=0.75,
                max_tokens=1024,
        )
        
        model_outputs = {}
        for instruction in tqdm(instructions):
            instruction_id = instruction["id"]
            eval_query = instruction["eval_query"]
            for update_id, instruction_topic in enumerate(instruction["text"]):
                prompt = prompt_template.replace("[eval_query]", eval_query).replace("[instruction_topic]", instruction_topic)
                messages = [{"role": "system", "content": preamble}, {"role": "user", "content": prompt}] if preamble != '' else [{"role": "user", "content": prompt}]
                response = model.chat(messages, gen_params)[0].outputs[0].text
                model_outputs[f"{instruction_id}.{update_id}"] = response
    
    else:
        ### Initialize the Cohere client
        if connection_mode == 'cohere':
            ### Generate instructions
            try:
                COHERE_API_KEY = os.environ["COHERE_API_KEY"]
            except KeyError:
                raise ValueError("You need to set the COHERE_API_KEY env variable!")
            client = cohere.Client(COHERE_API_KEY)
            
            i = 0
            model_outputs = {}
            for instruction in tqdm(instructions):
                instruction_id = instruction["id"]
                eval_query = instruction["eval_query"]
                for update_id, instruction_topic in enumerate(instruction["text"]):
                    prompt = prompt_template.replace("[eval_query]", eval_query).replace("[instruction_topic]", instruction_topic)
                    response = cohere_model_generate(
                            prompt, client, preamble, model_name="command-r-plus-08-2024", temperature=0.9, p=0.9
                    )
                    model_outputs[f"{instruction_id}.{update_id}"] = response
                    i += 1
                    if i % 10 == 0:
                        time.sleep(10)  # to avoid rate limiting
          
        else:
            ### Generate instructions
            if connection_mode == 'gpt':
                client = OpenAI()
            else:
                client = OpenAI(base_url=model_url, api_key=api_key)
            
            i = 0
            model_outputs = {}
            for instruction in tqdm(instructions):
                instruction_id = instruction["id"]
                eval_query = instruction["eval_query"]
                for update_id, instruction_topic in enumerate(instruction["text"]):
                    prompt = prompt_template.replace("[eval_query]", eval_query).replace("[instruction_topic]", instruction_topic)
                    response = openai_model_generate(
                            client, prompt, preamble, model_name=model_name
                    )
                    model_outputs[f"{instruction_id}.{update_id}"] = response
                    i += 1
                    if i % 10 == 0:
                        time.sleep(10)  # to avoid rate limiting
            
    model_name = model_name.split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}.json")
    with open(output_file, "a") as f:
        json.dump(model_outputs, f, indent=2)


def generate_model_output_session(dialogue_file, model_name, instruction_output_path, output_dir, connection_mode='cohere', n_gpus=2,
                                      model_url='http://localhost:12000/v1', api_key='a1b2c3d4e5', cache_path=None):
    """Generates the session and history model outputs.
    The already computed instruction-only output is included in the final output file"""
    with open(instruction_output_path, "r") as f:
        precomputed_instruction_model_output = json.load(f)

    # Load dialogue
    with open(dialogue_file, "r") as f:
        dialogue = json.load(f)
    dialogue_context = dialogue["context"]

    # Prompts
    preamble = f"""
    ## Task and Context
    You are {dialogue_context["mentee"]}, a new software engineer at {dialogue_context["company"]}. Your mentor {dialogue_context["mentor"]} has given you specific coding guidelines that you must follow.

    ## Style Guide
    Do not acknowledge. Only generate Python code and nothing else before or after. Do not explain the code. Do not ask for more information but directly give the answer.
    """
    session_prompt = f"This is a thread of dialogues between you and your mentor [mentor]:\n [session] \nBased on information provided, write a [eval_query]. Do not provide example usage. You must follow all the latest coding guidelines provided by your mentor, including any possible updates."
    
    if connection_mode not in ['cohere', 'openai', 'vllm']:
        cache_dir = Path(cache_path) if cache_path is not None else Path(__file__).absolute().parent.parent.parent.parent / 'cache'
        model = LLM(model_name, gpu_memory_utilization=0.95, enforce_eager=True, download_dir=str(cache_dir), tensor_parallel_size=n_gpus, dtype='auto', max_model_len=2048)
        gen_params = SamplingParams(
                temperature=1.0,
                top_k=5,
                top_p=0.75,
                max_tokens=1024,
        )
    
        # Generate model output
        model_outputs = []
        instruction_ids = dialogue["instructions"]
        for session_id, session in tqdm(enumerate(dialogue["sessions"])):
            session_eval_query = session["session_eval_query"]
    
            session_model_output = []
            instruction_model_output = []
            if session_eval_query:
                for eval in session_eval_query:
                    ## SESSION
                    prompt = (
                        session_prompt.replace("[mentor]", dialogue_context["mentor"])
                        .replace("[eval_query]", eval)
                        .replace("[session]", session["text"])
                    )
                    messages = [{"role": "system", "content": preamble}, {"role": "user", "content": prompt}] if preamble != '' else [{"role": "user", "content": prompt}]
                    response = model.chat(messages, gen_params)[0].outputs[0].text
                    session_model_output.append(response)
    
                ## INSTRUCTION
                for p, u in instruction_ids[session_id]:
                    response = precomputed_instruction_model_output[f"{p}.{u}"]
                    instruction_model_output.append(response)
    
            model_output = {
                "session_model_output": session_model_output,
                "instruction_model_output": instruction_model_output,
            }
            model_outputs.append(model_output)
    
        model_outputs = {"sessions": model_outputs}
    
    else:
        if connection_mode == 'cohere':
            # Generate model output
            try:
                COHERE_API_KEY = os.environ["COHERE_API_KEY"]
            except KeyError:
                raise ValueError("You need to set the COHERE_API_KEY env variable!")
            client = cohere.Client(COHERE_API_KEY)
            
            i = 0
            model_outputs = []
            instruction_ids = dialogue["instructions"]
            for session_id, session in tqdm(enumerate(dialogue["sessions"])):
                session_eval_query = session["session_eval_query"]
                
                session_model_output = []
                instruction_model_output = []
                if session_eval_query:
                    for eval in session_eval_query:
                        ## SESSION
                        prompt = (
                                session_prompt.replace("[mentor]", dialogue_context["mentor"])
                                .replace("[eval_query]", eval)
                                .replace("[session]", session["text"])
                        )
                        response = cohere_model_generate(prompt, client, preamble, model_name="command-r-plus-08-2024", temperature=0.9, p=0.9)
                        session_model_output.append(response)
                        i += 1
                        if i % 10 == 0:
                            time.sleep(10)  # to avoid rate limiting
                    
                    ## INSTRUCTION
                    for p, u in instruction_ids[session_id]:
                        response = precomputed_instruction_model_output[f"{p}.{u}"]
                        instruction_model_output.append(response)
                
                model_output = {
                        "session_model_output":     session_model_output,
                        "instruction_model_output": instruction_model_output,
                }
                model_outputs.append(model_output)
        
        else:
            # Generate model output
            if connection_mode == 'gpt':
                client = OpenAI()
            else:
                client = OpenAI(base_url=model_url, api_key=api_key)
            
            i = 0
            model_outputs = []
            instruction_ids = dialogue["instructions"]
            for session_id, session in tqdm(enumerate(dialogue["sessions"])):
                session_eval_query = session["session_eval_query"]
                
                session_model_output = []
                instruction_model_output = []
                if session_eval_query:
                    for eval in session_eval_query:
                        ## SESSION
                        prompt = (
                                session_prompt.replace("[mentor]", dialogue_context["mentor"])
                                .replace("[eval_query]", eval)
                                .replace("[session]", session["text"])
                        )
                        response = openai_model_generate(client, prompt, preamble, model_name=model_name)
                        session_model_output.append(response)
                        i += 1
                        if i % 10 == 0:
                            time.sleep(10)  # to avoid rate limiting
                    
                    ## INSTRUCTION
                    for p, u in instruction_ids[session_id]:
                        response = precomputed_instruction_model_output[f"{p}.{u}"]
                        instruction_model_output.append(response)
                
                model_output = {
                        "session_model_output":     session_model_output,
                        "instruction_model_output": instruction_model_output,
                }
                model_outputs.append(model_output)
    
        model_outputs = {"sessions": model_outputs}
    
    # Save model output
    model_name = model_name.split("/")[-1]
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    dialogue_id = os.path.basename(dialogue_file).split("_")[1].split(".")[0]
    output_file = os.path.join(output_dir, f"output_{dialogue_id}.json")
    with open(output_file, "w") as f:
        json.dump(model_outputs, f, indent=2)


def generate_model_output_history(dialogue_file, model_name, instruction_session_path, output_dir, connection_mode='cohere', n_gpus=2,
                                      model_url='http://localhost:12000/v1', api_key='a1b2c3d4e5', cache_path=None):
    """Generates the session and history model outputs.
    The already computed instruction-only output is included in the final output file"""
    model_name = model_name.split("/")[-1]
    with open(instruction_session_path, "r") as f:
        model_outputs = json.load(f)
    
    # Load dialogue
    with open(dialogue_file, "r") as f:
        dialogue = json.load(f)
    dialogue_context = dialogue["context"]
    
    # Prompts
    preamble = f"""
    ## Task and Context
    You are {dialogue_context["mentee"]}, a new software engineer at {dialogue_context["company"]}. Your mentor {dialogue_context["mentor"]} has given you specific coding guidelines that you must follow.

    ## Style Guide
    Do not acknowledge. Only generate Python code and nothing else before or after. Do not explain the code. Do not ask for more information but directly give the answer.
    """
    history_prompt = f"This is a thread of dialogues between you and your mentor [mentor]:\n [session] \nBased on information provided, write a [eval_query]. Do not provide example usage. You must follow all the latest coding guidelines provided by your mentor, including any possible updates."
    
    if connection_mode not in ['cohere', 'openai', 'vllm']:
        cache_dir = Path(cache_path) if cache_path is not None else Path(__file__).absolute().parent.parent.parent.parent / 'cache'
        model = LLM(model_name, gpu_memory_utilization=0.95, enforce_eager=True, download_dir=str(cache_dir), tensor_parallel_size=n_gpus, dtype='auto', max_model_len=2048)
        gen_params = SamplingParams(
                temperature=1.0,
                top_k=5,
                top_p=0.75,
                max_tokens=1024,
        )

        # Generate model output
        history_sessions = ""
        for session_id, session in tqdm(enumerate(dialogue["sessions"])):
            history_eval_query = session["history_eval_query"]
            history_sessions += f"\n\n Session {session_id} \n\n" + session["text"]
    
            history_model_output = []
            if session_id == len(dialogue["sessions"]) - 1:
                for eval in history_eval_query:
                    prompt = (
                        history_prompt.replace("[mentor]", dialogue_context["mentor"])
                        .replace("[eval_query]", eval)
                        .replace("[session]", history_sessions)
                    )
                    messages = [{"role": "system", "content": preamble}, {"role": "user", "content": prompt}] if preamble != '' else [{"role": "user", "content": prompt}]
                    response = model.chat(messages, gen_params)[0].outputs[0].text
                    history_model_output.append(response)
    
            model_output = {
                "history_model_output": history_model_output,
            }
            model_outputs["sessions"][session_id].update(model_output)
            
    else:
        if connection_mode == 'cohere':
            try:
                COHERE_API_KEY = os.environ["COHERE_API_KEY"]
            except KeyError:
                raise ValueError("You need to set the COHERE_API_KEY env variable!")
            client = cohere.Client(COHERE_API_KEY)
            
            i = 0
            history_sessions = ""
            for session_id, session in tqdm(enumerate(dialogue["sessions"])):
                history_eval_query = session["history_eval_query"]
                history_sessions += f"\n\n Session {session_id} \n\n" + session["text"]
        
                history_model_output = []
                if session_id == len(dialogue["sessions"]) - 1:
                    for eval in history_eval_query:
                        prompt = (
                            history_prompt.replace("[mentor]", dialogue_context["mentor"])
                            .replace("[eval_query]", eval)
                            .replace("[session]", history_sessions)
                        )
                        response = cohere_model_generate(prompt, client, preamble, model_name="command-r-plus-08-2024", temperature=0.9, p=0.9)
                        history_model_output.append(response)
                        i += 1
                        if i % 10 == 0:
                            time.sleep(10)  # to avoid rate limiting
        
                model_output = {
                    "history_model_output": history_model_output,
                }
                model_outputs["sessions"][session_id].update(model_output)
        
        else:
            if connection_mode == 'gpt':
                client = OpenAI()
            else:
                client = OpenAI(base_url=model_url, api_key=api_key)
                
            i = 0
            history_sessions = ""
            for session_id, session in tqdm(enumerate(dialogue["sessions"])):
                history_eval_query = session["history_eval_query"]
                history_sessions += f"\n\n Session {session_id} \n\n" + session["text"]
        
                history_model_output = []
                if session_id == len(dialogue["sessions"]) - 1:
                    for eval in history_eval_query:
                        prompt = (
                            history_prompt.replace("[mentor]", dialogue_context["mentor"])
                            .replace("[eval_query]", eval)
                            .replace("[session]", history_sessions)
                        )
                        response = openai_model_generate(client, prompt, preamble, model_name=model_name)
                        history_model_output.append(response)
                        i += 1
                        if i % 10 == 0:
                            time.sleep(10)  # to avoid rate limiting
        
                model_output = {
                    "history_model_output": history_model_output,
                }
                model_outputs["sessions"][session_id].update(model_output)
    
    # Save model output
    model_name = model_name.split("/")[-1]
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(instruction_session_path))
    with open(output_file, "w") as f:
        json.dump(model_outputs, f, indent=2)


def generate_model_output_instructions_chain(dialogue_file, model_name, output_dir, connection_mode='cohere', n_gpus=2,
                              model_url='http://localhost:12000/v1', api_key='a1b2c3d4e5', cache_path=None):
    """Output when a chain of instructions is provided as input."""
    model_generate = get_model_generate_function(model_name, temperature=0, p=None)

    model_name = model_name.split("/")[-1]

    # Load dialogue
    with open(dialogue_file, "r") as f:
        dialogue = json.load(f)

    # Prompts
    preamble = f"""
    ## Style Guide
    Do not acknowledge. Only generate Python code and nothing else before or after. Do not explain the code. Do not ask for more information but directly give the answer. 
    """
    prompt_template = f"This is a list of coding guidelines: [instruction_topic]. Some guidelines might have been updated. You must follow all the latest versions of the guidelines. Write a [eval_query]. Do not provide example usage."

    # Generate model output
    if connection_mode not in ['cohere', 'openai', 'vllm']:
        cache_dir = Path(cache_path) if cache_path is not None else Path(__file__).absolute().parent.parent.parent.parent / 'cache'
        model = LLM(model_name, gpu_memory_utilization=0.95, enforce_eager=True, download_dir=str(cache_dir), tensor_parallel_size=n_gpus, dtype='auto', max_model_len=2048)
        gen_params = SamplingParams(
                temperature=1.0,
                top_k=5,
                top_p=0.75,
                max_tokens=1024,
        )
        
        model_outputs = []
        instruction_topics = []
        for session_id, session in tqdm(enumerate(dialogue["sessions"])):
            history_eval_query = session["history_eval_query"]
            for type, instruction_topic in zip(session["type"], session["topic"]):
                if "instruction" in type:
                    instruction_topics.append(instruction_topic)
    
            model_output = []
            if session_id == len(dialogue["sessions"]) - 1:
                instruction_topics_str = ", ".join(instruction_topics)
                for eval in history_eval_query:
                    prompt = prompt_template.replace("[eval_query]", eval).replace("[instruction_topic]", instruction_topics_str)
                    messages = [{"role": "system", "content": preamble}, {"role": "user", "content": prompt}] if preamble != '' else [{"role": "user", "content": prompt}]
                    response = model.chat(messages, gen_params)[0].outputs[0].text
                    model_output.append(response)
    
            model_output = {"instructions_only_model_output": model_output}
            model_outputs.append(model_output)
    
        model_outputs = {"sessions": model_outputs}
        
    else:
        if connection_mode == 'cohere':
            try:
                COHERE_API_KEY = os.environ["COHERE_API_KEY"]
            except KeyError:
                raise ValueError("You need to set the COHERE_API_KEY env variable!")
            client = cohere.Client(COHERE_API_KEY)
            
            i = 0
            model_outputs = []
            instruction_topics = []
            for session_id, session in tqdm(enumerate(dialogue["sessions"])):
                history_eval_query = session["history_eval_query"]
                for type, instruction_topic in zip(session["type"], session["topic"]):
                    if "instruction" in type:
                        instruction_topics.append(instruction_topic)
                
                model_output = []
                if session_id == len(dialogue["sessions"]) - 1:
                    instruction_topics_str = ", ".join(instruction_topics)
                    for eval in history_eval_query:
                        prompt = prompt_template.replace("[eval_query]", eval).replace("[instruction_topic]", instruction_topics_str)
                        response = cohere_model_generate(prompt, client, preamble, model_name="command-r-plus-08-2024", temperature=0.9, p=0.9)
                        model_output.append(response)
                        i += 1
                        if i % 10 == 0:
                            time.sleep(10)  # to avoid rate limiting
                
                model_output = {"instructions_only_model_output": model_output}
                model_outputs.append(model_output)
            
            model_outputs = {"sessions": model_outputs}
            
        else:
            if connection_mode == 'gpt':
                client = OpenAI()
            else:
                client = OpenAI(base_url=model_url, api_key=api_key)
                
            i = 0
            model_outputs = []
            instruction_topics = []
            for session_id, session in tqdm(enumerate(dialogue["sessions"])):
                history_eval_query = session["history_eval_query"]
                for type, instruction_topic in zip(session["type"], session["topic"]):
                    if "instruction" in type:
                        instruction_topics.append(instruction_topic)
                
                model_output = []
                if session_id == len(dialogue["sessions"]) - 1:
                    instruction_topics_str = ", ".join(instruction_topics)
                    for eval in history_eval_query:
                        prompt = prompt_template.replace("[eval_query]", eval).replace("[instruction_topic]", instruction_topics_str)
                        response = openai_model_generate(client, prompt, preamble, model_name=model_name)
                        model_output.append(response)
                        i += 1
                        if i % 10 == 0:
                            time.sleep(10)  # to avoid rate limiting
                
                model_output = {"instructions_only_model_output": model_output}
                model_outputs.append(model_output)
            
            model_outputs = {"sessions": model_outputs}
            
    # Save model output
    model_name = model_name.split("/")[-1]
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    dialogue_id = os.path.basename(dialogue_file).split("_")[1].split(".")[0]
    output_file = os.path.join(output_dir, f"output_{dialogue_id}.json")
    with open(output_file, "w") as f:
        json.dump(model_outputs, f, indent=2)


def generate_model_output_rag(dialogue_file, model_name, output_dir, connection_mode='cohere', n_gpus=2,
                              model_url='http://localhost:12000/v1', api_key='a1b2c3d4e5', cache_path=None):
    """RAG"""
    # Load dialogue
    with open(dialogue_file, "r") as f:
        dialogue = json.load(f)
    dialogue_context = dialogue["context"]

    # Prompts
    preamble = f"""
    ## Task and Context
    You are {dialogue_context["mentee"]}, a new software engineer at {dialogue_context["company"]}. Your mentor {dialogue_context["mentor"]} has given you specific coding guidelines that you must follow.

    ## Style Guide
    Do not acknowledge. Only generate Python code and nothing else before or after. Do not explain the code. Do not ask for more information but directly give the answer.
    """
    rag_prompt = f"This is a thread of dialogues between you and your mentor [mentor]:\n [session] \nBased on information provided, write a [eval_query]. Do not provide example usage. You must follow all the latest coding guidelines provided by your mentor, including any possible updates."

    # Get number of instructions sessions
    num_instruction_sessions = 0
    for p in dialogue["instructions"]:
        if p != [-1]:
            num_instruction_sessions += 1

    # Generate model output
    if connection_mode not in ['cohere', 'openai', 'vllm']:
        cache_dir = Path(cache_path) if cache_path is not None else Path(__file__).absolute().parent.parent.parent.parent / 'cache'
        model = LLM(model_name, gpu_memory_utilization=0.95, enforce_eager=True, download_dir=str(cache_dir), tensor_parallel_size=n_gpus, dtype='auto', max_model_len=2048)
        gen_params = SamplingParams(
                temperature=1.0,
                top_k=5,
                top_p=0.75,
                max_tokens=1024,
        )
        
        model_outputs = []
        history_sessions_list = []
        for session_id, session in tqdm(enumerate(dialogue["sessions"])):
            history_eval_query = session["history_eval_query"]
            history_sessions_list.append(session["text"])
    
            rag_model_output = []
            if session_id == len(dialogue["sessions"]) - 1:
                for eval in history_eval_query:
                    ### RAG
                    relevant_chunks = perform_retrieval(history_sessions_list, eval, num_instruction_sessions)
                    prompt = (
                        rag_prompt.replace("[mentor]", dialogue_context["mentor"])
                        .replace("[eval_query]", eval)
                        .replace("[session]", relevant_chunks)
                    )
                    messages = [{"role": "system", "content": preamble}, {"role": "user", "content": prompt}] if preamble != '' else [{"role": "user", "content": prompt}]
                    response = model.chat(messages, gen_params)[0].outputs[0].text
                    rag_model_output.append(response)
    
            model_output = {"rag_model_output": rag_model_output}
            model_outputs.append(model_output)
    
        model_outputs = {"sessions": model_outputs}
    
    else:
        if connection_mode == 'cohere':
            try:
                COHERE_API_KEY = os.environ["COHERE_API_KEY"]
            except KeyError:
                raise ValueError("You need to set the COHERE_API_KEY env variable!")
            
            client = cohere.Client(COHERE_API_KEY)
            model_outputs = []
            history_sessions_list = []
            i = 0
            for session_id, session in tqdm(enumerate(dialogue["sessions"])):
                history_eval_query = session["history_eval_query"]
                history_sessions_list.append(session["text"])
                
                rag_model_output = []
                if session_id == len(dialogue["sessions"]) - 1:
                    for eval in history_eval_query:
                        ### RAG
                        relevant_chunks = perform_retrieval(history_sessions_list, eval, num_instruction_sessions)
                        prompt = (
                                rag_prompt.replace("[mentor]", dialogue_context["mentor"])
                                .replace("[eval_query]", eval)
                                .replace("[session]", relevant_chunks)
                        )
                        response = cohere_model_generate(prompt, client, preamble, model_name="command-r-plus-08-2024", temperature=0.9, p=0.9)
                        rag_model_output.append(response)
                        i += 1
                        if i % 10 == 0:
                            time.sleep(10)  # to avoid rate limiting
                
                model_output = {"rag_model_output": rag_model_output}
                model_outputs.append(model_output)
            
            model_outputs = {"sessions": model_outputs}
        
        else:
            if connection_mode == 'gpt':
                client = OpenAI()
            else:
                client = OpenAI(base_url=model_url, api_key=api_key)
            
            model_outputs = []
            history_sessions_list = []
            i = 0
            for session_id, session in tqdm(enumerate(dialogue["sessions"])):
                history_eval_query = session["history_eval_query"]
                history_sessions_list.append(session["text"])
                
                rag_model_output = []
                if session_id == len(dialogue["sessions"]) - 1:
                    for eval in history_eval_query:
                        ### RAG
                        relevant_chunks = perform_retrieval(history_sessions_list, eval, num_instruction_sessions)
                        prompt = (
                                rag_prompt.replace("[mentor]", dialogue_context["mentor"])
                                .replace("[eval_query]", eval)
                                .replace("[session]", relevant_chunks)
                        )
                        response = openai_model_generate(client, prompt, preamble, model_name=model_name)
                        rag_model_output.append(response)
                        i += 1
                        if i % 10 == 0:
                            time.sleep(10)  # to avoid rate limiting
                
                model_output = {"rag_model_output": rag_model_output}
                model_outputs.append(model_output)
            
            model_outputs = {"sessions": model_outputs}
    
    # Save model output
    model_name = model_name.split("/")[-1]
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    dialogue_id = os.path.basename(dialogue_file).split("_")[1].split(".")[0]
    output_file = os.path.join(output_dir, f"output_{dialogue_id}.json")
    with open(output_file, "w") as f:
        json.dump(model_outputs, f, indent=2)


def generate_model_output_no_dialogue(dialogue_file, model_name, output_dir, connection_mode='cohere', n_gpus=2,
                              model_url='http://localhost:12000/v1', api_key='a1b2c3d4e5', cache_path=None):
    """Output when no dialogue is provided."""
    model_name = model_name.split("/")[-1]

    # Load dialogue
    with open(dialogue_file, "r") as f:
        dialogue = json.load(f)

    # Prompts
    preamble = f"""
    ## Style Guide
    Do not acknowledge. Only generate Python code and nothing else before or after. Do not explain the code. Do not ask for more information but directly give the answer. 
    """
    prompt_template = f"Write a [eval_query]. Do not provide example usage."

    # Generate model output
    if connection_mode not in ["cohere", "gpt", "vllm"]:
        cache_dir = Path(cache_path) if cache_path is not None else Path(__file__).absolute().parent.parent.parent.parent / 'cache'
        model = LLM(model_name, gpu_memory_utilization=0.95, enforce_eager=True, download_dir=str(cache_dir), tensor_parallel_size=n_gpus, dtype='auto', max_model_len=2048)
        gen_params = SamplingParams(
                temperature=1.0,
                top_k=5,
                top_p=0.75,
                max_tokens=1024,
        )
        
        model_outputs = []
        for session_id, session in tqdm(enumerate(dialogue["sessions"])):
            history_eval_query = session["history_eval_query"]
    
            model_output = []
            if session_id == len(dialogue["sessions"]) - 1:
                ## history
                for eval in history_eval_query:
                    prompt = prompt_template.replace("[eval_query]", eval)
                    messages = [{"role": "system", "content": preamble}, {"role": "user", "content": prompt}] if preamble != '' else [{"role": "user", "content": prompt}]
                    response = model.chat(messages, gen_params)[0].outputs[0].text
                    model_output.append(response)
    
            model_output = {"no_dialogue_model_output": model_output}
            model_outputs.append(model_output)
    
        model_outputs = {"sessions": model_outputs}
    
    else:
        if connection_mode == 'cohere':
            try:
                COHERE_API_KEY = os.environ["COHERE_API_KEY"]
            except KeyError:
                raise ValueError("You need to set the COHERE_API_KEY env variable!")
            
            client = cohere.Client(COHERE_API_KEY)
            i = 0
            model_outputs = []
            for session_id, session in tqdm(enumerate(dialogue["sessions"])):
                history_eval_query = session["history_eval_query"]
                
                model_output = []
                if session_id == len(dialogue["sessions"]) - 1:
                    ## history
                    for eval in history_eval_query:
                        prompt = prompt_template.replace("[eval_query]", eval)
                        response = cohere_model_generate(prompt, client, preamble, model_name="command-r-plus-08-2024", temperature=0.9, p=0.9)
                        model_output.append(response)
                        i += 1
                        if i % 10 == 0:
                            time.sleep(10)  # to avoid rate limiting
                
                model_output = {"no_dialogue_model_output": model_output}
                model_outputs.append(model_output)
            
            model_outputs = {"sessions": model_outputs}
            
        else:
            if connection_mode == 'gpt':
                client = OpenAI()
            else:
                client = OpenAI(base_url=model_url, api_key=api_key)
            
            i = 0
            model_outputs = []
            for session_id, session in tqdm(enumerate(dialogue["sessions"])):
                history_eval_query = session["history_eval_query"]
                
                model_output = []
                if session_id == len(dialogue["sessions"]) - 1:
                    ## history
                    for eval in history_eval_query:
                        prompt = prompt_template.replace("[eval_query]", eval)
                        response = openai_model_generate(client, prompt, preamble, model_name=model_name)
                        model_output.append(response)
                        i += 1
                        if i % 10 == 0:
                            time.sleep(10)  # to avoid rate limiting
                
                model_output = {"no_dialogue_model_output": model_output}
                model_outputs.append(model_output)
            
            model_outputs = {"sessions": model_outputs}
    
    # Save model output
    model_name = model_name.split("/")[-1]
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    dialogue_id = os.path.basename(dialogue_file).split("_")[1].split(".")[0]
    output_file = os.path.join(output_dir, f"output_{dialogue_id}.json")
    with open(output_file, "w") as f:
        json.dump(model_outputs, f, indent=2)


if __name__ == "__main__":
    fire.Fire(generate_model_output_history)
