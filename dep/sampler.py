import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

def load_jsonl(file_path):
    return pd.read_json(file_path, lines=True)

def preprocess_data(df, keys):
    # Filter the dataframe based on the specified keys
    filtered_df = df[keys]
    
    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse=False)
    encoded_data = encoder.fit_transform(filtered_df.select_dtypes(include=['object', 'category']))
    
    # Combine encoded data with numerical data
    numerical_data = filtered_df.select_dtypes(include=['int64', 'float64'])
    combined_data = np.hstack([encoded_data, numerical_data.to_numpy()])
    
    return combined_data

def sample_diverse_entries(data, num_samples=1000, num_clusters=200):
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    
    # Initialize an empty list to hold sampled indices
    sampled_indices = []
    
    # Sample from each cluster
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        num_to_sample = min(len(cluster_indices), max(1, num_samples // num_clusters))
        sampled_indices.extend(np.random.choice(cluster_indices, num_to_sample, replace=False))
    
    # If we sampled less than num_samples, sample randomly from the entire dataset to fill the gap
    additional_samples_needed = num_samples - len(sampled_indices)
    if additional_samples_needed > 0:
        all_indices = np.arange(data.shape[0])
        remaining_indices = np.setdiff1d(all_indices, sampled_indices)
        sampled_indices.extend(np.random.choice(remaining_indices, additional_samples_needed, replace=False))
    
    return sampled_indices

# Load JSONL file
file_path = 'personas.jsonl'
df = load_jsonl(file_path)

# Specify the keys for diversity sampling
keys = [
    "Sex", "Year_of_Birth", "Age", "Is_Immigrant", "Is_Immigrant_Mother", 
    "Is_Immigrant_Father", "Country_of_Residence", "Country_of_Birth", 
    "Country_of_Birth_Mother", "Country_of_Birth_Father", "n_People_in_Household", 
    "Live_with_Parents", "Marital_Status", "n_Children", "Education", "Education_Spouse", 
    "Education_Mother", "Education_Father", "Employment_Status", "Employment_Status_Spouse", 
    "Occupational_Group", "Occupational_Group_Spouse", "Occupational_Group_Father", 
    "Works_for", "Chief_Wage_Earner", "Last_Year_Savings", "Self_Assessed_Social_Class", 
    "Income_Level", "Religious_Denomination"
]

# Preprocess data
preprocessed_data = preprocess_data(df, keys)

# Sample diverse entries
sampled_indices = sample_diverse_entries(preprocessed_data)

# Retrieve sampled rows
sampled_df = df.iloc[sampled_indices]

# Specify the path to save the sampled JSONL file
output_jsonl_path = 'personas-sampled.jsonl'

# Save the DataFrame as JSONL
sampled_df.to_json(output_jsonl_path, orient='records', lines=True)