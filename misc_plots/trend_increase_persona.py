import plotly.graph_objects as go
import numpy as np

# Values and colors for each line
values = [
    "Self-Direction", "Security", "Hedonism", "Conformity", "Universalism",
    "Power", "Tradition", "Stimulation", "Benevolence", "Achievement"
]
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Data for the line graph
data_points = [5, 25, 50]
data_means = {
    "Self-Direction": [0.7059, 0.7702, 0.7762],
    "Security": [1, 1, 1],
    "Hedonism": [0.0784, 0.0766, 0.0584],
    "Conformity": [0.1667, 0.1802, 0.1762],
    "Universalism": [0.6994, 0.8363, 0.8288],
    "Power": [0.0686, 0.0676, 0.0324],
    "Tradition": [0.4118, 0.4977, 0.454],
    "Stimulation": [0, 0, 0],
    "Benevolence": [0.5294, 0.6171, 0.5968],
    "Achievement": [0.4706, 0.3694, 0.3503]
}
data_variances = {
    "Self-Direction": [0.0087, 0.0012, 0.0026],
    "Security": [0.0024, 0.0014, 0.0002],
    "Hedonism": [0.0022, 0.0026, 0.0008],
    "Conformity": [0.0044, 0.0002, 0.0016],
    "Universalism": [0.017, 0.0004, 0.0005],
    "Power": [0.001, 0.0009, 0.0004],
    "Tradition": [0.0123, 0.0044, 0.0009],
    "Stimulation": [0.0009, 0.0009, 0],
    "Benevolence": [0.0023, 0.0026, 0.0001],
    "Achievement": [0.0026, 0.0028, 0.0028]
}

# Create figure
fig = go.Figure()

# Add traces for each value
for value, color in zip(values, colors):
    # Calculate the mean and variance for each report
    mean_values = data_means[value]
    variance_values = data_variances[value]
    
    # Calculate error (standard deviation)
    error = np.sqrt(variance_values)
    
    # Add the main line for the mean values
    fig.add_trace(go.Scatter(
        x=data_points, y=mean_values,
        mode='lines',
        name=value,
        line=dict(color=color)
    ))
    
    # Add a trace for the upper bound of the error (mean + error)
    fig.add_trace(go.Scatter(
        x=data_points, y=np.array(mean_values) + error,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ))
    
    # Add a trace for the lower bound of the error (mean - error)
    fig.add_trace(go.Scatter(
        x=data_points, y=np.array(mean_values) - error,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.1)',
        fill='tonexty',
        showlegend=False
    ))

# Update the layout of the figure
fig.update_layout(
    title="Values across Reports with Shaded Error Bands",
    xaxis_title="Report Number",
    yaxis_title="Mean Value",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    paper_bgcolor='white',
    plot_bgcolor='white',
    hovermode="x unified"
)

# Save the figure to a HTML file

# Save the figure as an HTML file
html_path = "plots/trend_increase_persona.html"
fig.write_html(html_path)
