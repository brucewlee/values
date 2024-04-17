import json

def jsonl_to_html_with_expandable_entries(input_jsonl_path, output_html_path):
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Expandable JSONL Viewer</title>
    <style>
        .prompt { margin-bottom: 20px; cursor: pointer; }
        .prompt-header { font-weight: bold; }
        .prompt-content { display: none; margin-left: 20px; }
    </style>
</head>
<body>
    <h1>Expandable JSONL Data View</h1>
    """

    with open(input_jsonl_path, 'r') as jsonl_file:
        for index, line in enumerate(jsonl_file):
            entry = json.loads(line)
            html_content += f'''
<div class="prompt" onclick="toggleVisibility('prompt-content-{index}')">
    <div class="prompt-header">Prompt {index + 1}</div>
</div>
<div id="prompt-content-{index}" class="prompt-content">
    <pre>{json.dumps(entry, indent=4)}</pre>
</div>
'''

    # Adding JavaScript function for toggling visibility
    html_content += """
<script>
function toggleVisibility(id) {
    var x = document.getElementById(id);
    if (x.style.display === "none") {
        x.style.display = "block";
    } else {
        x.style.display = "none";
    }
}
</script>
</body>
</html>
"""

    with open(output_html_path, 'w') as html_file:
        html_file.write(html_content)

# Convert your JSONL file to an HTML file with expandable content
jsonl_to_html_with_expandable_entries('benchmark/intermediaries/prompts-response.jsonl', 'output_html_file.html')
