import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import plotly.graph_objects as go

def load_jsonl(file_path):
    return pd.read_json(file_path, lines=True)

def preprocess_data(df, keys):
    filtered_df = df[keys]
    encoder = OneHotEncoder(sparse=False)
    encoded_data = encoder.fit_transform(filtered_df.select_dtypes(include=['object', 'category']))
    numerical_data = filtered_df.select_dtypes(include=['int64', 'float64'])
    combined_data = np.hstack([encoded_data, numerical_data.to_numpy()])
    return combined_data

def interactive_elbow_method(inertias, range_of_clusters):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range_of_clusters), y=inertias, mode='lines+markers', name='Inertia'))
    fig.update_layout(title='Elbow Method for Optimal Clusters',
                      xaxis_title='Number of Clusters',
                      yaxis_title='Inertia',
                      hovermode='closest')
    fig.show()

def interactive_silhouette_scores(silhouette_scores, range_of_clusters):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range_of_clusters), y=silhouette_scores, mode='lines+markers', name='Silhouette Score'))
    fig.update_layout(title='Silhouette Scores for Various Clusters',
                      xaxis_title='Number of Clusters',
                      yaxis_title='Silhouette Score',
                      hovermode='closest')
    fig.show()

def find_optimal_clusters(data, max_clusters=10):
    inertias = []
    silhouette_scores = []
    range_of_clusters = range(2, max_clusters + 1)

    for n_clusters in range_of_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    interactive_elbow_method(inertias, range_of_clusters)
    interactive_silhouette_scores(silhouette_scores, range_of_clusters)

    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because range starts at 2
    return optimal_clusters

# Sample Usage
file_path = 'benchmark/personas.jsonl'  # Update this path to your actual file path
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
]  # Update this list with the keys you're interested in
preprocessed_data = preprocess_data(df, keys)
optimal_clusters = find_optimal_clusters(preprocessed_data, max_clusters=10)  # Adjust `max_clusters` as needed

print(f"Optimal number of clusters: {optimal_clusters}")
