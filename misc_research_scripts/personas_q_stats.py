import pandas as pd

# Load the JSONL file
df = pd.read_json('benchmark/personas.jsonl', lines=True)

# Select the question columns, which are those that start with 'Q'
question_columns = [col for col in df.columns if col.startswith('Q')]

# Create a DataFrame to store the counts for each option for each question
stats_df = pd.DataFrame(columns=[1, 2, 3, 4, 5], index=question_columns)

# Count the occurrences of each option for each question
for col in question_columns:
    stats_df.loc[col] = df[col].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).tolist()

# Save the statistics as a JSONL file
output_file = 'benchmark/useful_statistics/personas_q.jsonl'
stats_df.reset_index().rename(columns={'index': 'Question'}).to_json(output_file, orient='records', lines=True)
