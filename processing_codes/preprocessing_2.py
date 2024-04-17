import pandas as pd

# Load your DataFrame
df = pd.read_csv('Values_processed.csv')

# Convert entire DataFrame to string to safely check for "-" prefix
df = df.astype(str)

# Replace values that start with "-" with 'N/A' in specified columns
exceptions = ['Education_Spouse', 'Employment_Status_Spouse']
for column in exceptions:
    df[column] = df[column].apply(lambda x: 'N/A' if x.startswith('-') else x)

# Function to check if any cell in a row starts with "-", ignoring exception columns
def has_negative_value(row, exception_columns):
    for column, value in row.items():
        if column not in exception_columns and isinstance(value, str) and value.startswith('-'):
            return True
    return False

# Apply the function to filter out rows
filtered_df = df[~df.apply(lambda row: has_negative_value(row, exceptions), axis=1)]

# Optionally, convert numeric columns back to their appropriate types here
# Example: filtered_df['Age'] = pd.to_numeric(filtered_df['Age'], errors='coerce')

# After filtering, save the DataFrame to a new CSV file
filtered_df.to_csv('filtered_values.csv', index=False)

print("Filtered and processed DataFrame saved to 'filtered_values.csv'.")
