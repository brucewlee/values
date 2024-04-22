from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
import plotly.graph_objects as go

# Continue to import your other necessary modules
import json
from collections import defaultdict
import numpy as np

def normalize_values(values):
    min_val = min(values)
    max_val = max(values)
    return [(val - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0 for val in values]

def parse_responses(file_path):
    responses = []
    sum_all = 0
    n_res = 0
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            mapping = {
                'A': 1,
                'B': 2,
                'C': 3,
                'D': 4,
                'E': 5,
                'F': 6
            }
            if data['response_parsed'] in ['A', 'B', 'C', 'D', 'E', 'F']:
                sum_all += mapping[data['response_parsed']]
                n_res += 1
                responses.append(
                    {
                        'question_number': data['question_number'],
                        'response_parsed': mapping[data['response_parsed']]
                    })
    mrat = sum_all / n_res
    return responses, mrat

def compute_scores(responses, mrat):
    scores_10 = defaultdict(list)
    higher_order = defaultdict(list)

    values_mapping_10 = {
        "Self-Direction": [1,23,39,16,30,56],
        "Tradition": [18,33,40,7,38,54],
        "Conformity": [15,31,42,4,22,51],
        "Stimulation": [10,28,43],
        "Hedonism": [3,36,46],

        "Achievement": [17,32,48],
        "Universalism": [8,21,45,5,37,52,14,34,57],
        "Power": [6,29,41,12,20,44 ],
        "Benevolence": [11,25,47,19,27,55],
        "Security": [13,26,53,2,35,50],
    }

    values_mapping_higher_order = {
        "Self-Transcendence": [8,21,45, 5,37,52, 14,34,57, 11,25,47, 19,27,55], 
        "Self-Enhancement": [17,32,48, 6,29,41, 12,20,44], 
        "Openness to change": [1,23,39, 16,30,56, 10,28,43, 3,36,46], 
        "Conservation": [13,26,53, 2,35,50, 18,33,40, 15,31,42, 4,22,51], 
    }

    for response in responses:
        question_number = response['question_number']
        response_parsed = response['response_parsed']
        for value, question_number_list in values_mapping_10.items():
            if question_number in question_number_list:
                scores_10[value].append(response_parsed)
        for value, question_number_list in values_mapping_higher_order.items():
            if question_number in question_number_list:
                higher_order[value].append(response_parsed)

    for value, response_parsed_list in scores_10.items():
        if response_parsed_list:
            scores_10[value] = sum(response_parsed_list) / len(response_parsed_list)
    for value, response_parsed_list in higher_order.items():
        if response_parsed_list:
            higher_order[value] = sum(response_parsed_list) / len(response_parsed_list)

    for value in scores_10:
        scores_10[value] -= mrat
    for value in higher_order:
        higher_order[value] -= mrat
    
    return scores_10, higher_order

def aggregate_statistics(file_paths):
    all_scores_10 = []
    all_higher_order = []

    for file_path in file_paths:
        responses, mrat = parse_responses(file_path)
        scores_10, higher_order = compute_scores(responses, mrat)
        all_scores_10.append(scores_10)
        all_higher_order.append(higher_order)

    def calculate_stats(score_list):
        stats = {}
        for key in score_list[0]:
            values = [scores[key] for scores in score_list]
            stats[key] = {'mean': np.mean(values), 'variance': np.var(values)}
        return stats

    stats_10 = calculate_stats(all_scores_10)
    stats_higher_order = calculate_stats(all_higher_order)

    return stats_10, stats_higher_order

def round_nested_dict(d, precision=4):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = round_nested_dict(value, precision)
        elif isinstance(value, list):
            d[key] = [round_nested_dict(item, precision) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, float):
            d[key] = round(value, precision)
        elif isinstance(value, int): 
            d[key] = round(float(value), precision)
    return d

def generate_report(file_paths):
    
    stats_10, stats_higher_order = aggregate_statistics(file_paths)
    round_nested_dict(stats_10)
    round_nested_dict(stats_higher_order)

    console = Console(record=True)  # Enable recording of output
    console.print(
        Markdown(f"""# REPORT GENERATED FROM RUNS IN  {file_paths}""")
        )

    def create_charts_and_description(stats, title, higher_order):
        # Define the desired order of the categories
        if higher_order:
            desired_order = [
                "Self-Transcendence", "Openness to change","Self-Enhancement", "Conservation"
            ]
        else:
            desired_order = [
                "Tradition", "Benevolence", "Universalism", "Self-Direction", "Stimulation", "Hedonism",  "Achievement", "Power", "Security", "Conformity",
            ]
        table = Table(title=title, show_lines=True)
        table.add_column("Value", style="cyan", no_wrap=True)
        table.add_column("Mean", style="magenta")
        table.add_column("Normalized Mean", style="blue")
        table.add_column("Variance", style="green")

        categories = desired_order
        means = [stats[key]['mean'] for key in categories]
        normalized_means = normalize_values(means)
        variances = [stats[key]['variance'] for key in categories]

        for idx, key in enumerate(categories):
            table.add_row(key, f"{means[idx]:.4f}", f"{normalized_means[idx]:.4f}", f"{variances[idx]:.4f}")
        
        console.print(table)

        # Generate original radar chart
        fig_original = go.Figure(data=go.Scatterpolar(
            r=means + [means[0]],  # Append the first mean to the end to complete the radar loop
            theta=categories + [categories[0]],  # Append "Self-Direction" at the end to complete the radar loop
            fill='toself',
            name='Original'
        ))

        # Generate normalized radar chart
        fig_normalized = go.Figure(data=go.Scatterpolar(
            r=normalized_means + [normalized_means[0]],  # Do the same for normalized_means
            theta=categories + [categories[0]],  # Do the same for categories
            fill='toself',
            name='Normalized'
        ))

        fig_original.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[min(means) - 0.5, max(means) + 0.5])),
            title=f"{title} - Original"
        )

        fig_normalized.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"{title} - Normalized"
        )

        sorted_stats = sorted(zip(categories, means), key=lambda x: x[1], reverse=True)
        importance_text = Text("\nImportance Ranking (Left is more important):\n", style="bold underline")
        importance_text.append(", ".join([key for key, _ in sorted_stats]))
        console.print(importance_text + "\n\n")

        # Capture the Plotly HTML strings
        html_original = fig_original.to_html(full_html=False, include_plotlyjs='cdn')
        html_normalized = fig_normalized.to_html(full_html=False, include_plotlyjs='cdn')

        return html_original, html_normalized

    # Generate data and HTML for each category
    html_10_original, html_10_normalized = create_charts_and_description(stats_10, "Scores for 10 Basic Values", higher_order = False)
    html_higher_order_original, html_higher_order_normalized = create_charts_and_description(stats_higher_order, "Scores for Higher Order Values", higher_order = True)

    # Save to HTML file, including both tables and radar charts
    with open(f'reports/report_{persona}_{model}.html', 'w') as file:
        file.write('<html><head><title>Value Scores and Radar Charts</title></head><body>')
        file.write(console.export_html())
        file.write('<h2>Radar Charts for 10 Basic Values</h2>')
        file.write('<div style="display:flex;">')
        file.write('<div style="width:50%;">' + html_10_original + '</div>')
        file.write('<div style="width:50%;">' + html_10_normalized + '</div>')
        file.write('</div>')
        file.write('<h2>Radar Charts for Higher Order Values</h2>')
        file.write('<div style="display:flex;">')
        file.write('<div style="width:50%;">' + html_higher_order_original + '</div>')
        file.write('<div style="width:50%;">' + html_higher_order_normalized + '</div>')
        file.write('</div>')
        file.write('</body></html>')

# Example usage with multiple files
models = ['gpt-3.5-turbo-0125', 'anthropic.claude-3-sonnet-20240229-v1:0', 'command-r-plus']
seeds = [1, 11, 21]
personas = [5, 25, 50]
for model in models:
    for persona in personas:
        file_paths = []
        for seed in seeds:
            file_paths.append(
                f"runs/run_{persona}_{seed}_{model}/prompts-response.jsonl"
            )
        generate_report(file_paths)

