import json
from collections import Counter
import math

# Function to read a JSONL file and return a list of dictionaries
def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Function to calculate the entropy of a distribution
def calculate_entropy(distribution):
    total_responses = sum(distribution.values())
    if total_responses == 0:
        return 0
    entropy = 0
    for count in distribution.values():
        probability = count / total_responses
        if probability > 0:  # To avoid log(0)
            entropy -= probability * math.log2(probability)
    return entropy


# Read the files
prompts_responses = read_jsonl_file('../runs/run_50_1_anthropic.claude-3-sonnet-20240229-v1:0/prompts-response.jsonl')
personas = read_jsonl_file('../benchmark/personas.jsonl')
questions = read_jsonl_file('../benchmark/questions.jsonl')
# Transforming personas into a dictionary for easier lookup
personas_dict = {persona['Index']: persona for persona in personas}

# Analyzing responses and calculating entropy
response_stats = {}
entropies = []

for response in prompts_responses:
    question_idx = response['Question_idx']
    # Limiting to questions Q1 through Q6
    if question_idx > 6:
        continue
    persona_idx = response['Persona_idx']
    response_key = response['Response_Parsed']
    
    if question_idx not in response_stats:
        response_stats[question_idx] = {'model_responses': []}
    
    # Append model responses to stats
    response_stats[question_idx]['model_responses'].append(response_key)

# Calculate entropy for each question's model response distribution
for question_idx, stats in response_stats.items():
    model_response_count = Counter(stats['model_responses'])
    entropy = calculate_entropy(model_response_count)
    entropies.append(entropy)
    print(f"Question {question_idx}: Model Response Entropy = {entropy}")

# Calculate and print the FPT score (average entropy)
fpt_score = sum(entropies) / len(entropies) if entropies else 0
print(f"\nAverage Flexibility in Perspective-Taking (FPT) Score for Q1 through Q6: {fpt_score}")