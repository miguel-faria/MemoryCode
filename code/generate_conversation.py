import json
import os

import cohere
import fire
from generation_utils import cohere_model_generate
from tqdm import tqdm


def generate_conversation(prompt_file, template_dir):
    """Generates a conversation given a prompt then populates the corresponding template conversation file."""
    with open(prompt_file, "r") as f:
        prompt_data = json.load(f)
    conversation_id = os.path.basename(prompt_file).split("_")[1].split(".")[0]
    conversation_file = os.path.join(template_dir, f"conversation_{conversation_id}.json")
    assert os.path.exists(conversation_file), f"Template file {conversation_file} does not exist!"
    with open(conversation_file, "r") as f:
        template = json.load(f)

    ### Initialize the Cohere client
    try:
        COHERE_API_KEY = os.environ["COHERE_API_KEY"]
    except KeyError:
        raise ValueError("You need to set the COHERE_API_KEY env variable!")
    client = cohere.Client(COHERE_API_KEY)

    ### Generate conversation
    preamble = prompt_data["preamble"]
    for session, prompt in tqdm(zip(template["sessions"], prompt_data["prompts"])):
        session_text = cohere_model_generate(
            prompt, client, preamble, model_name="command-r-plus", temperature=0.9, p=0.9
        )
        session["text"] = session_text

    ### save conversation dataset
    with open(conversation_file, "w") as f:
        json.dump(template, f, indent=2)


if __name__ == "__main__":
    fire.Fire(generate_conversation)
