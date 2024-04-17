import pandas as pd
import numpy as np
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
from models import ChatGPT, Claude3Sonnet, Mistral8x7BInst, LLaMA2_70BChat, BaseModel

def install_traceback():
    install()

def initialize_console():
    console = Console(record=True)
    return console

def set_experiment_parameters(
    personas_per_question, seed, model, console
    ):
    console.print(
        Markdown(f"""# EXPERIMENT PARAMETERS \n 1. *PERSONAS_PER_QUESTION*: {personas_per_question} \n 2. *SEED*: {seed} \n3. *MODEL*: {BaseModel().judge_model}""")
        )
    # Constructing the directory path
    dir_path = f"runs/run_{personas_per_question}_{seed}_{model.model}/"
    # Check if the directory exists, if not create it
    os.makedirs(dir_path, exist_ok=True)
    # File paths adjusted to use the new directory path
    prompts_jsonl_path = os.path.join(dir_path, 'prompts.jsonl')
    prompts_response_jsonl_path = os.path.join(dir_path, 'prompts-response.jsonl')
    console_output_path = os.path.join(dir_path, f"ask_question_terminal.html")
    # Set a seed for reproducibility
    np.random.seed(seed)
    return dir_path, prompts_jsonl_path, prompts_response_jsonl_path, console_output_path

def generate_prompts(
    personas_per_question, seed, model, console, prompts_jsonl_path
    ):
        console.print(
        Markdown(f"""# Preparing Prompts \n 1. *PERSONAS*: benchmark/personas.jsonl \n 2. *QUESTIONS*: benchmark/questions.jsonl""")
        )
        personas_df = pd.read_json('benchmark/personas.jsonl', lines=True)
        questions_df = pd.read_json('benchmark/questions.jsonl', lines=True)
        combined_data = []
        
        # Iterate through each question
        for _, question_row in questions_df.iterrows():
            q_idx = question_row['Index']
            options = ['OptionA', 'OptionB', 'OptionC', 'OptionD', 'OptionE']
            valid_options = [o for o in options if pd.notna(question_row[o + '_Value'])]

            # New logic starts here
            # Group options by their value to ensure equal representation
            option_values = question_row[[o + '_Value' for o in valid_options]].dropna().unique()
            value_to_options = {value: [o for o in valid_options if question_row[o + '_Value'] == value] for value in option_values}

            samples_per_value = personas_per_question // len(option_values)
            extra_samples = personas_per_question % len(option_values)

            for value, options_for_value in value_to_options.items():
                total_samples_for_value = samples_per_value + (1 if extra_samples > 0 else 0)
                extra_samples -= 1 if extra_samples > 0 else 0

                samples_per_option_for_value = total_samples_for_value // len(options_for_value)
                extra_samples_for_value = total_samples_for_value % len(options_for_value)

                for option in options_for_value:
                    option_samples = samples_per_option_for_value + (1 if extra_samples_for_value > 0 else 0)
                    extra_samples_for_value -= 1 if extra_samples_for_value > 0 else 0

                    # Now sample personas based on the corrected logic
                    selected_personas = personas_df[personas_df[f'Q{q_idx}'] == question_row[option + '_Value']]
                    sampled_personas = selected_personas.sample(n=option_samples, replace=len(selected_personas) < option_samples)

                    for _, persona_row in sampled_personas.iterrows():
                        prompt = "Let's role-play. I will ask you a question and you must give me an answer. I want you to act as the person described below. Think from the person's perspective. \n\n"
                        prompt += f"{persona_row['Description']} \n\n"
                        prompt += "Use the given information to answer the question below. Your response should always point to a specific letter option: "
                        if 'OptionA' in valid_options:
                            prompt += f"'A. {question_row['OptionA']}'"
                        if 'OptionB' in valid_options:
                            prompt += f" or 'B. {question_row['OptionB']}'"
                        if 'OptionC' in valid_options:
                            prompt += f" or 'C. {question_row['OptionC']}'"
                        if 'OptionD' in valid_options:
                            prompt += f" or 'D. {question_row['OptionD']}'"
                        if 'OptionE' in valid_options:
                            prompt += f" or 'E. {question_row['OptionE']}'"
                        prompt += f". You must always choose one option. You are role-playing, and you are not allowed to decline to choose an option. \n\n"
                        prompt += f"{question_row['Question']} \n\n"
                        if 'OptionA' in valid_options:
                            prompt += f"A. {question_row['OptionA']} \n"
                        if 'OptionB' in valid_options:
                            prompt += f"B. {question_row['OptionB']} \n"
                        if 'OptionC' in valid_options:
                            prompt += f"C. {question_row['OptionC']} \n"
                        if 'OptionD' in valid_options:
                            prompt += f"D. {question_row['OptionD']} \n"
                        if 'OptionE' in valid_options:
                            prompt += f"E. {question_row['OptionE']} \n"

                        combined_entry = {
                            'Prompt': prompt,
                            'Question_idx': q_idx,
                            'Persona_idx': persona_row['Index']
                        }
                        combined_data.append(combined_entry)

        # To save the combined_data as JSONL with the json module
        with open(prompts_jsonl_path, 'w') as outfile:
            for entry in combined_data:
                json.dump(entry, outfile)
                outfile.write('\n')

def retrieve_responses(
    model, prompts_jsonl_path, prompts_response_jsonl_path, console, console_output_path
    ):
    console.print(
        Markdown(f"""# Retrieve Responses \n 1. *RESPONDENT*: {model.model} \n 2. *RESPONSE PARSER*: {BaseModel().judge_model}""")
        )

    def load_prompts(infile_path, step=1):
        prompts_data = []
        with open(infile_path, 'r') as infile:
            for i, line in enumerate(infile):
                if i % step == 0:  # Select every 50th entry
                    prompts_data.append(json.loads(line))
        return prompts_data

        # Open the file for writing processed prompts with responses
    with open(prompts_response_jsonl_path, 'w') as outfile:
        # Load data into a list first
        
        prompts_data = load_prompts(prompts_jsonl_path, step=1)

        with Progress(console=console, transient=True) as progress:
            task = progress.add_task("Retrieving Responses ...", total=len(prompts_data))
            for prompt_entry in prompts_data:
                prompt_text = prompt_entry['Prompt']
                
                # Call the API (or the simulation of it) with the prompt
                api_response = model.respond(prompt_text)     
                prompt_entry['Response'] = api_response[0] 
                prompt_entry['Response_Parsed'] = api_response[1]

                # Write the updated prompt_entry back to 'prompts-response.jsonl'
                json.dump(prompt_entry, outfile)
                outfile.write('\n')
                outfile.flush()

                console.print(
                    f"\nSaved Response for a Question-Persona Pair: Question {prompt_entry['Question_idx']} and Persona {prompt_entry['Persona_idx']}"
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
        }
    return args.personas_per_question, args.seed, model_dict[args.model]

def main():
    install_traceback()

    console = initialize_console()

    personas_per_question, seed, model = parse_arguments()

    dir_path, prompts_jsonl_path, prompts_response_jsonl_path, console_output_path = set_experiment_parameters(
        personas_per_question, seed, model, console
        )
        
    generate_prompts(
        personas_per_question, seed, model, console, prompts_jsonl_path
        )

    retrieve_responses(
        model, prompts_jsonl_path, prompts_response_jsonl_path, console, console_output_path
        )

if __name__ == "__main__":
    main()
