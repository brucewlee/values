import json
import csv

# Define the path to your JSONL file
jsonl_path = '/Users/bruce/Walnut Research/values/runs/Claude3Sonnet/scenario-response.jsonl'
tsv_path = '/Users/bruce/Walnut Research/values/runs/Claude3Sonnet/scenario-response.tsv'

# Fields to extract from the JSONL file
fields = ['Context', 'Action 1', 'Action 2', 'Action 1 - Value (A)', 'Action 2 - Value (B)', 'A/B Response Parsed', 'Repeat Response Parsed', 'Compare Response Parsed']

# Read the JSONL file and write to TSV
with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file, open(tsv_path, 'w', newline='', encoding='utf-8') as tsv_file:
    writer = csv.DictWriter(tsv_file, fieldnames=fields, delimiter='\t')
    writer.writeheader()  # Write header to the TSV file

    for line in jsonl_file:
        try:
            data = json.loads(line)
            
            # Extract the required fields
            row = {
                'Context': data['Context'],
                'Action 1': data['Action 1'],
                'Action 2': data['Action 2'],
                'Action 1 - Value (A)': data['Action 1 - Value'],
                'Action 2 - Value (B)': data['Action 2 - Value'],
                'A/B Response Parsed': data['A/B Response Parsed'],
                'Repeat Response Parsed': data['Repeat Response Parsed'],
                'Compare Response Parsed': data['Compare Response Parsed']
            }
            
            # Write the extracted data to the TSV file
            writer.writerow(row)
        except:
            pass

print("Data has been successfully written to the TSV file.")
