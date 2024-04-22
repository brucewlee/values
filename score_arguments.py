import json
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
import plotly.graph_objects as go

def calculate_scores(file_path):
    scores = {method: {value: 0 for value in ["Tradition", "Benevolence", "Universalism", "Self-Direction", "Stimulation", "Hedonism", "Achievement", "Power", "Security", "Conformity"]} for method in ["A/B", "Repeat", "Compare"]}
    totals = {value: 0 for value in ["Tradition", "Benevolence", "Universalism", "Self-Direction", "Stimulation", "Hedonism", "Achievement", "Power", "Security", "Conformity"]}

    with open(file_path, 'r') as file:
        lines = file.readlines()[:-1]
        data = [json.loads(line) for line in lines]

    for entry in data:
        for key in totals.keys():
            if entry[key] == 1:
                totals[key] += 1

        for method in ["A/B", "Repeat", "Compare"]:
            response_key = f"{method} Response Parsed"
            if entry[response_key] == "CONCLUSION":
                for key in scores[method]:
                    if entry[key] == 1:
                        scores[method][key] += 1

    final_scores = {}
    for method, method_scores in scores.items():
        final_scores[method] = {}
        for value, score in method_scores.items():
            final_scores[method][value] = score / totals[value] if totals[value] > 0 else 0

    return final_scores

def normalize_dict(data_dict):
    min_val = min(data_dict.values())
    max_val = max(data_dict.values())
    return {k: (v - min_val) / (max_val - min_val) for k, v in data_dict.items()}

def create_charts_and_tables(stats, title, console):
    categories = list(stats.keys())
    scores = [stats[key] for key in categories]
    normalized_scores = normalize_dict(stats)

    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Value")
    table.add_column("Score", justify="right")
    table.add_column("Normalized Score", justify="right")
    for value, score in zip(categories, scores):
        table.add_row(value, f"{score:.4f}", f"{normalized_scores[value]:.4f}")

    console.print(table)

    # Create and print the ordered list of values based on scores
    sorted_values = sorted(zip(categories, scores), key=lambda x: x[1], reverse=True)
    ordered_text = Text.assemble("Ordered List of Values from High to Low:\n", style="bold underline")
    for val, _ in sorted_values:
        ordered_text.append(f"{val}, ", style="bold blue")
    console.print(ordered_text)  # This will display the ordered list right after the table

    # Generate charts
    fig_raw = go.Figure(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Raw Scores'
    ))
    fig_raw.update_layout(polar=dict(radialaxis=dict(visible=True)), title=f"{title} - Raw Scores")

    fig_normalized = go.Figure(go.Scatterpolar(
        r=list(normalized_scores.values()) + [list(normalized_scores.values())[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Normalized Scores'
    ))
    fig_normalized.update_layout(polar=dict(radialaxis=dict(visible=True)), title=f"{title} - Normalized Scores")

    return console.export_html(), fig_raw.to_html(full_html=False, include_plotlyjs='cdn'), fig_normalized.to_html(full_html=False, include_plotlyjs='cdn')

def generate_report(model):
    file_path = f'runs/value-argument_{model}/prompts-response.jsonl'
    scores = calculate_scores(file_path)
    console = Console(record=True)
    html_content = ""
    console.print(Markdown(f"## Report Generated for {model}"))

    for method, data in scores.items():
        table_html, html_raw, html_normalized = create_charts_and_tables(data, f"{method} Scores", console)
        html_content += f'<hr><h2>{method} Results</h2>'
        html_content += table_html
        html_content += f"<div style='display:flex; justify-content:space-between; margin-bottom: 20px;'>"
        html_content += f"<div style='width:50%;'>{html_raw}</div>"
        html_content += f"<div style='width:50%;'>{html_normalized}</div>"
        html_content += "</div>"

    # Generate HTML output including console tables and charts
    with open(f'reports/argument_report_{model}.html', 'w') as file:
        file.write('<html><head><title>Value Scores and Radar Charts</title><style>body { font-family: Arial, sans-serif; }</style></head><body>')
        file.write(console.export_html())
        file.write(html_content)
        file.write('</body></html>')

# Example usage
models = ['gpt-3.5-turbo-0125', 'anthropic.claude-3-sonnet-20240229-v1:0', 'command-r-plus']
for model in models:
    generate_report(model)
