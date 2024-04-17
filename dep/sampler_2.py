import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def load_jsonl(file_path):
    return pd.read_json(file_path, lines=True)

def preprocess_data(df, keys):
    filtered_df = df[keys]
    encoder = OneHotEncoder(sparse=False)
    encoded_data = encoder.fit_transform(filtered_df.select_dtypes(include=['object', 'category']))
    numerical_data = filtered_df.select_dtypes(include=['int64', 'float64'])
    combined_data = np.hstack([encoded_data, numerical_data.to_numpy()])
    return combined_data

def find_optimal_clusters(data, max_clusters=100):
    inertias = []
    silhouette_scores = []
    range_of_clusters = range(2, max_clusters + 20)

    for n_clusters in range_of_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    # Plot Inertia to find the elbow
    plt.figure(figsize=(12, 6))
    plt.plot(range_of_clusters, inertias, '-o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Plot Silhouette Scores
    plt.figure(figsize=(12, 6))
    plt.plot(range_of_clusters, silhouette_scores, '-o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()
    
    # You might need to visually inspect the plots to choose the best cluster
    # Alternatively, programmatically choose the number with the highest silhouette score or the elbow point for inertia
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because range starts at 2
    return optimal_clusters

def sample_diverse_entries(data, num_samples=1000):
    num_clusters = find_optimal_clusters(data)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    sampled_indices = []

    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        num_to_sample = min(len(cluster_indices), max(1, num_samples // num_clusters))
        sampled_indices.extend(np.random.choice(cluster_indices, num_to_sample, replace=False))

    additional_samples_needed = num_samples - len(sampled_indices)
    if additional_samples_needed > 0:
        all_indices = np.arange(data.shape[0])
        remaining_indices = np.setdiff1d(all_indices, sampled_indices)
        sampled_indices.extend(np.random.choice(remaining_indices, additional_samples_needed, replace=False))

    return sampled_indices

# Load, preprocess, and sample as before
file_path = 'personas.jsonl'
df = load_jsonl(file_path)
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
preprocessed_data = preprocess_data(df, keys)
sampled_indices = sample_diverse_entries(preprocessed_data)
sampled_df = df.iloc[sampled_indices]
output_jsonl_path = 'personas-sampled.jsonl'
sampled_df.to_json(output_jsonl_path, orient='records', lines=True)
