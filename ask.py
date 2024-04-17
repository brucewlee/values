import pandas as pd
import numpy as np
import random
import json
from tqdm import tqdm
import os, argparse
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress
import rich
from rich.traceback import install
install()
console = Console(record=True)
from api_keys import set_api_keys
set_api_keys()
from models import ChatGPT, Claude3Sonnet, Mistral8x7BInst, LLaMA2_70BChat, BaseModel, CommandRPlus
from persona import generate_persona_description

def install_traceback():
    install()

def initialize_console():
    console = Console(record=True)
    return console

def set_experiment_parameters(
    personas_per_question, seed, model, console, save_path = 'runs/'
    ):
    console.print(
        Markdown(f"""# EXPERIMENT PARAMETERS \n 1. *PERSONAS_PER_QUESTION*: {personas_per_question} \n 2. *SEED*: {seed} \n3. *MODEL*: {BaseModel().judge_model}""")
        )
    # Constructing the directory path
    dir_path = f"{save_path}/run_{personas_per_question}_{seed}_{model.model}/"
    # Check if the directory exists, if not create it
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(dir_path, exist_ok=True)
    prompts_response_jsonl_path = os.path.join(dir_path, 'prompts-response.jsonl')
    console_output_path = os.path.join(dir_path, f"ask_question_terminal.html")
    # Set a seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    return dir_path, prompts_response_jsonl_path, console_output_path

def generate_prompts(
    personas_per_question, seed, model, console
    ):
    console.print(
    Markdown(f"""# Preparing Prompts""")
    )
    questions_df = pd.read_json('benchmark/questions.jsonl', lines=True)
    combined_data = []
    personas = generate_persona_description(personas_per_question)
    
    # Iterate through each question
    for _, question_row in questions_df.iterrows():
        q_idx = question_row['question_number']
        options = question_row['options']

        for persona in personas:
            prompt = "Let's role-play. I will ask you a question and you must give me an answer. I want you to act as the person described below. Think from the person's perspective. \n\n"
            prompt += f"{persona['description']} \n\n"
            prompt += "Use the given information to answer the question below. Your response should always point to a specific letter option.\n\n"
            prompt += f"Read the statement and think about how much that person is or is not like you.\n\n"
            prompt += f"STATEMENT: {question_row['statement']}\n"
            prompt += f"A. {options[0]} \n B. {options[1]} \n C. {options[2]} \n D. {options[3]} \n E. {options[4]} \n F. {options[5]} \n\n"

            combined_entry = {
                'prompt': prompt,
                'question_number': q_idx,
                'persona': persona
            }
            combined_data.append(combined_entry)
    
    return combined_data

def retrieve_responses(
    model, prompts_response_jsonl_path, console, console_output_path, combined_data
    ):
    console.print(
        Markdown(f"""# Retrieve Responses \n 1. *RESPONDENT*: {model.model} \n 2. *RESPONSE PARSER*: {BaseModel().judge_model}""")
        )

        # Open the file for writing processed prompts with responses
    with open(prompts_response_jsonl_path, 'w') as outfile:
        # Load data into a list first

        with Progress(console=console, transient=True) as progress:
            task = progress.add_task("Retrieving Responses ...", total=len(combined_data))
            for prompt_entry in combined_data:
                prompt_text = prompt_entry['prompt']
                
                # Call the API (or the simulation of it) with the prompt
                api_response = model.respond(prompt_text)     
                prompt_entry['response'] = api_response[0] 
                prompt_entry['response_parsed'] = api_response[1]

                # Write the updated prompt_entry back to 'prompts-response.jsonl'
                json.dump(prompt_entry, outfile)
                outfile.write('\n')
                outfile.flush()

                console.print(
                    f"\nSaved Response for a Question-Persona Pair: Question {prompt_entry['question_number']} and Persona {prompt_entry['persona']}"
                    )
                progress.advance(task)
                console.save_html(console_output_path, clear=False)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run experiments with personas and questions.')
    parser.add_argument('-p', '--personas_per_question', type=int, default=50, help='Number of personas per question')
    parser.add_argument('-s', '--seed', type=int, default=1, help='Seed for random number generation')
    parser.add_argument('-m', '--model', type=str, default='Claude3Sonnet', help='Model to use for generating responses')
    args = parser.parse_args()
    model_dict = {
        'Claude3Sonnet': Claude3Sonnet(), 
        'ChatGPT': ChatGPT(),
        'Mistral8x7BInst': Mistral8x7BInst(),
        'LLaMA2-70BChat': LLaMA2_70BChat(),
        'CommandRPlus': CommandRPlus()
        }
    return args.personas_per_question, args.seed, model_dict[args.model]

def main():
    install_traceback()

    console = initialize_console()

    personas_per_question, seed, model = parse_arguments()

    dir_path, prompts_response_jsonl_path, console_output_path = set_experiment_parameters(
        personas_per_question, seed, model, console
        )
        
    combined_data = generate_prompts(
        personas_per_question, seed, model, console
        )

    retrieve_responses(
        model, prompts_response_jsonl_path, console, console_output_path, combined_data
        )

if __name__ == "__main__":
    main()
