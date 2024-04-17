import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm
from sklearn.metrics import silhouette_score

# Assuming 'data.jsonl' is the file with 20,000 lines
file_path = 'benchmark/personas.jsonl'

# Load the data
df = pd.read_json(file_path, lines=True)

# Preprocessing Pipeline
preprocessing_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('scaler', StandardScaler(with_mean=False))  # Use with_mean=False to handle sparse matrix
])

# Selecting relevant keys for clustering
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

# Encoding and scaling the data
preprocessed_data = preprocessing_pipeline.fit_transform(df[keys])

# Perform PCA for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(preprocessed_data.toarray())

# Function to calculate silhouette score for different number of clusters
def calculate_silhouette(data, range_clusters):
    silhouette_scores = []
    
    for n_clusters in tqdm(range_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Number of clusters: {n_clusters}, Silhouette score: {silhouette_avg}")
    
    return silhouette_scores


# Define the range of clusters to try
range_clusters = range(2, 10000, 1000)  # For example, trying from 2 to 10 clusters

# Calculate silhouette scores for the range of clusters
silhouette_scores = calculate_silhouette(preprocessed_data, range_clusters)

# Find the optimal number of clusters (highest silhouette score)
optimal_n_clusters = range_clusters[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_n_clusters}")

# Run KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(preprocessed_data)

# Visualize the PCA-reduced clusters
fig = px.scatter(x=pca_data[:, 0], y=pca_data[:, 1], color=cluster_labels.astype(str))
fig.update_layout(
    title="PCA-reduced Clusters Visualization",
    xaxis_title="PCA Component 1",
    yaxis_title="PCA Component 2",
    legend_title="Cluster"
)
# Save the figure to a file
pio.write_image(fig, 'cluster_visualization.svg')

# Sample representative data points
# Assuming we want to sample 10 points per cluster
representative_indices = []
for i in np.unique(cluster_labels):
    cluster_indices = np.where(cluster_labels == i)[0]
    representative_indices.extend(np.random.choice(cluster_indices, 1, replace=False))

representative_df = df.iloc[representative_indices]

# Visualize the sampled points in PCA-reduced space
sampled_pca_data = pca.transform(preprocessing_pipeline.transform(representative_df[keys]).toarray())
sampled_fig = px.scatter(x=sampled_pca_data[:, 0], y=sampled_pca_data[:, 1])
sampled_fig.update_layout(
    title="Sampled Points Visualization",
    xaxis_title="PCA Component 1",
    yaxis_title="PCA Component 2"
)
# Save the sampled points figure to a file
pio.write_image(sampled_fig, 'sampled_points_visualization.svg')

# Save the sampled data for publication
representative_df.to_json('sampled_data.jsonl', orient='records', lines=True)

# Note: To improve the visual quality for publication, consider using vector graphic formats (e.g., SVG).
