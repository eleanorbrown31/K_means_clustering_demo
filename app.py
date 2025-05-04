import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set page title
st.title("K-means Clustering - Machine Learning in Action")

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = np.array([])
if 'k' not in st.session_state:
    st.session_state.k = 3
if 'max_iterations' not in st.session_state:
    st.session_state.max_iterations = 10
if 'current_iteration' not in st.session_state:
    st.session_state.current_iteration = 0
if 'centroids' not in st.session_state:
    st.session_state.centroids = np.array([])
if 'labels' not in st.session_state:
    st.session_state.labels = np.array([])
if 'inertia_history' not in st.session_state:
    st.session_state.inertia_history = []
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame({'X': [], 'Y': []})

# Function to generate random clustered data
def generate_random_data():
    num_clusters = st.session_state.k
    points_per_cluster = 20
    spread = 50
    new_data = []
    
    for i in range(num_clusters):
        center_x = 100 + np.random.random() * 400
        center_y = 100 + np.random.random() * 400
        
        for j in range(points_per_cluster):
            offset_x = (np.random.random() - 0.5) * spread
            offset_y = (np.random.random() - 0.5) * spread
            new_data.append([center_x + offset_x, center_y + offset_y])
    
    st.session_state.data = np.array(new_data)
    st.session_state.df = pd.DataFrame(new_data, columns=['X', 'Y'])
    st.session_state.centroids = np.array([])
    st.session_state.labels = np.array([])
    st.session_state.current_iteration = 0
    st.session_state.inertia_history = []

# Function to initialize centroids randomly
def initialize_centroids():
    if len(st.session_state.data) == 0:
        return
    
    kmeans = KMeans(
        n_clusters=st.session_state.k,
        max_iter=1,
        n_init=1,
        random_state=42
    )
    
    kmeans.fit(st.session_state.data)
    
    st.session_state.centroids = kmeans.cluster_centers_
    st.session_state.labels = kmeans.labels_
    st.session_state.current_iteration = 0
    
    st.session_state.inertia_history = [{"iteration": 0, "inertia": kmeans.inertia_}]

# Perform one iteration of k-means
def run_iteration():
    if len(st.session_state.data) == 0 or len(st.session_state.centroids) == 0:
        return False
    
    kmeans = KMeans(
        n_clusters=st.session_state.k,
        max_iter=2,
        n_init=1,
        init=st.session_state.centroids,
        random_state=42
    )
    
    kmeans.fit(st.session_state.data)
    
    centroid_change = np.sum((kmeans.cluster_centers_ - st.session_state.centroids) ** 2)
    changed = centroid_change > 0.001
    
    st.session_state.centroids = kmeans.cluster_centers_
    st.session_state.labels = kmeans.labels_
    st.session_state.current_iteration += 1
    
    st.session_state.inertia_history.append({
        "iteration": st.session_state.current_iteration, 
        "inertia": kmeans.inertia_
    })
    
    return changed

# Function to draw the current state
def draw_visualization():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [
        '#ff6b6b', '#48dbfb', '#1dd1a1', '#feca57', '#54a0ff',
        '#5f27cd', '#ff9ff3', '#00d2d3', '#576574', '#222f3e'
    ]
    
    if len(st.session_state.data) > 0:
        if len(st.session_state.labels) > 0:
            for i in range(st.session_state.k):
                cluster_points = st.session_state.data[st.session_state.labels == i]
                if len(cluster_points) > 0:
                    ax.scatter(
                        cluster_points[:, 0], 
                        cluster_points[:, 1], 
                        c=colors[i % len(colors)],
                        alpha=0.7,
                        s=50,
                        label=f'Cluster {i+1}'
                    )
            
            ax.scatter(
                st.session_state.centroids[:, 0],
                st.session_state.centroids[:, 1],
                c='black',
                s=100,
                marker='x',
                label='Centroids'
            )
            
            ax.legend(loc='upper right')
        else:
            ax.scatter(
                st.session_state.data[:, 0],
                st.session_state.data[:, 1],
                c='#999',
                alpha=0.7,
                s=50
            )
    
    if len(st.session_state.data) > 0:
        x_min, y_min = np.min(st.session_state.data, axis=0) - 20
        x_max, y_max = np.max(st.session_state.data, axis=0) + 20
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 400)
    
    ax.set_title("K-means Clustering Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    return fig

# Create a simple two-part layout
st.write("## Data Input")

# Simple data generation
if st.button("Generate Random Data"):
    generate_random_data()
    st.success("Random data generated successfully!")

# Display current data
if len(st.session_state.data) > 0:
    st.write(f"Current data points: {len(st.session_state.data)}")
    st.dataframe(st.session_state.df.head(10))

# Divider
st.markdown("---")

# Visualization and controls
col1, col2 = st.columns([2, 1])

with col1:
    st.write("## Clustering Visualization")
    
    if len(st.session_state.data) > 0:
        fig = draw_visualization()
        st.pyplot(fig)
    else:
        st.info("Generate random data first to see visualization.")

with col2:
    st.write("## Algorithm Controls")
    
    if len(st.session_state.data) > 0:
        # K value
        st.session_state.k = st.slider(
            "Number of clusters (k)", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.k
        )
        
        # Max iterations
        st.session_state.max_iterations = st.number_input(
            "Max iterations", 
            min_value=1, 
            max_value=50, 
            value=st.session_state.max_iterations
        )
        
        # Initialize button
        if st.button("Initialize Clustering"):
            initialize_centroids()
        
        # Step button
        step_disabled = (len(st.session_state.centroids) == 0 or 
                      st.session_state.current_iteration >= st.session_state.max_iterations)
        
        if st.button("Run One Iteration", disabled=step_disabled):
            run_iteration()
        
        # Status information
        st.write("### Status")
        st.write(f"Iteration: **{st.session_state.current_iteration} / {st.session_state.max_iterations}**")
        
        if len(st.session_state.inertia_history) > 0:
            current_inertia = st.session_state.inertia_history[-1]["inertia"]
            st.write(f"Inertia: **{current_inertia:.2f}** (lower is better)")

# Display learning progress chart if we have history data
if len(st.session_state.inertia_history) > 1:
    st.write("## Learning Progress")
    
    history_df = pd.DataFrame(st.session_state.inertia_history)
    st.line_chart(history_df.set_index("iteration")["inertia"])
    
    st.caption(
        "The chart shows how the algorithm improves with each iteration. "
        "Lower inertia means better clustering."
    )

# Educational section
st.markdown("---")
st.write("## How K-means Clustering Works")

st.markdown("""
### The Algorithm:
1. Initialize k random cluster centers (centroids)
2. Assign each data point to the nearest centroid
3. Recalculate centroids as the average of all points in the cluster
4. Repeat steps 2-3 until centroids no longer move significantly

### Applications:
- Customer segmentation in marketing
- Image compression and colour quantization
- Anomaly detection in cybersecurity
- Document clustering for topic modelling
- Genetic clustering in biology
""")