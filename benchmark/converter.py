import csv
import json

# Define the path to your TSV file
tsv_paths = ['value_decision_scenarios - model 1.tsv']
jsonl_path = 'scenarios.jsonl'

# Open the JSONL file outside the loop to append data from all TSV files
with open(jsonl_path, mode='a', encoding='utf-8') as jsonl_file:  # Use 'a' mode for appending
    for tsv_path in tsv_paths:
        # Read the TSV file
        with open(tsv_path, mode='r', newline='', encoding='utf-8') as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter='\t')
            
            for row in reader:
                # Here, you can reintroduce any logic to filter which rows to include
                if row.get('include?', '1') == '1':  # Assuming 'include?' column exists and default to include if missing
                    # Write the updated row to the JSONL file
                    jsonl_file.write(json.dumps(row) + '\n')

print("Conversion to JSONL complete with additional keys.")
