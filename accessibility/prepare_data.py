import json
import os
import glob
from collections import Counter

def load_question_statements(jsonl_path):
    """Load question statements from a JSONL file."""
    statements = {}
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            question_number = item['question_number']
            statement = item['statement']
            statements[question_number] = statement
    return statements


def jsonl_to_array(jsonl_path):
    """Read a JSONL file and return a list of Python dicts."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def calculate_response_counts(data):
    """Calculate the model's response distribution for each question."""
    response_stats = {}
    for item in data:
        question_idx = item['question_number']
        response_key = item['response_parsed']
        
        if question_idx not in response_stats:
            response_stats[question_idx] = []
        
        response_stats[question_idx].append(response_key)
    
    # Convert counts to the format suitable for JavaScript visualization
    response_counts = {q: dict(Counter(responses)) for q, responses in response_stats.items()}
    return response_counts

def generate_js_file(datasets, response_counts, output_path='data.js'):
    """Generate a JavaScript file defining the datasets and response counts as JS objects."""
    with open(output_path, 'w', encoding='utf-8') as js_file:
        js_file.write("var dataSets = ")
        json.dump(datasets, js_file, indent=4)
        js_file.write(";\n\nvar dataSetsResponseCounts = ")
        json.dump(response_counts, js_file, indent=4)
        js_file.write(";\n")

def main():
    runs_directory = '../runs/'
    questions_file = '../benchmark/questions.jsonl'  # Path to the questions.jsonl file
    datasets = {}
    datasets_response_counts = {}

    # Load question statements
    question_statements = load_question_statements(questions_file)

    # Iterate over each "prompts-response.jsonl" file in the runs directory
    for jsonl_file in glob.glob(f'{runs_directory}**/prompts-response.jsonl', recursive=True):
        # Extract a dataset name from the path
        dataset_name = os.path.basename(os.path.dirname(jsonl_file))
        # Convert the JSONL file to a Python list
        data = jsonl_to_array(jsonl_file)

        datasets[dataset_name] = data
        # Calculate response counts
        response_counts = calculate_response_counts(data)
        datasets_response_counts[dataset_name] = response_counts

    # Generate the JavaScript file including the question statements
    generate_js_file_with_statements(datasets, datasets_response_counts, question_statements)

def generate_js_file_with_statements(datasets, response_counts, question_statements, output_path='data.js'):
    """Generate a JavaScript file including datasets, response counts, and question statements."""
    with open(output_path, 'w', encoding='utf-8') as js_file:
        js_file.write("var dataSets = ")
        json.dump(datasets, js_file, indent=4, sort_keys=True)
        js_file.write(";\n\nvar dataSetsResponseCounts = ")
        json.dump(response_counts, js_file, indent=4, sort_keys=True)
        js_file.write(";\n\nvar questionStatements = ")
        json.dump(question_statements, js_file, indent=4, sort_keys=True)
        js_file.write(";\n")


if __name__ == "__main__":
    main()
