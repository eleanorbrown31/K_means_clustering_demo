import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

# Set page title and configuration
st.set_page_config(page_title="K-means Clustering - Machine Learning in Action", layout="wide")
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
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'animation_speed' not in st.session_state:
    st.session_state.animation_speed = 0.5  # seconds

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
    st.session_state.centroids = np.array([])
    st.session_state.labels = np.array([])
    st.session_state.current_iteration = 0
    st.session_state.inertia_history = []

# Function to clear all data
def clear_data():
    st.session_state.data = np.array([])
    st.session_state.centroids = np.array([])
    st.session_state.labels = np.array([])
    st.session_state.current_iteration = 0
    st.session_state.inertia_history = []
    st.session_state.is_running = False

# Function to initialize centroids randomly
def initialize_centroids():
    if len(st.session_state.data) == 0:
        return
    
    # Create KMeans object
    kmeans = KMeans(
        n_clusters=st.session_state.k,
        max_iter=1,  # Only initialize
        n_init=1,
        random_state=42
    )
    
    # Fit to get initial centroids
    kmeans.fit(st.session_state.data)
    
    # Store centroids and labels
    st.session_state.centroids = kmeans.cluster_centers_
    st.session_state.labels = kmeans.labels_
    st.session_state.current_iteration = 0
    
    # Calculate initial inertia
    initial_inertia = calculate_inertia(st.session_state.data, st.session_state.labels, st.session_state.centroids)
    st.session_state.inertia_history = [{"iteration": 0, "inertia": initial_inertia}]

# Calculate inertia (sum of squared distances to centroids)
def calculate_inertia(points, labels, centroids):
    total_distance = 0
    
    for i in range(len(centroids)):
        cluster_points = points[labels == i]
        if len(cluster_points) > 0:
            centroid = centroids[i]
            distances = np.sum((cluster_points - centroid) ** 2, axis=1)
            total_distance += np.sum(distances)
    
    return total_distance

# Perform one iteration of k-means
def run_iteration():
    if len(st.session_state.data) == 0 or len(st.session_state.centroids) == 0:
        return False
    
    # Create KMeans object with current centroids as init
    kmeans = KMeans(
        n_clusters=st.session_state.k,
        max_iter=2,  # One iteration from current state
        n_init=1,
        init=st.session_state.centroids,
        random_state=42
    )
    
    # Fit for one more iteration
    kmeans.fit(st.session_state.data)
    
    # Check if centroids changed significantly
    centroid_change = np.sum((kmeans.cluster_centers_ - st.session_state.centroids) ** 2)
    changed = centroid_change > 0.001
    
    # Update state
    st.session_state.centroids = kmeans.cluster_centers_
    st.session_state.labels = kmeans.labels_
    st.session_state.current_iteration += 1
    
    # Calculate and store inertia
    new_inertia = kmeans.inertia_
    st.session_state.inertia_history.append({
        "iteration": st.session_state.current_iteration, 
        "inertia": new_inertia
    })
    
    return changed

# Main layout
# Create two columns for the main content
col1, col2 = st.columns([2, 1])

# Left column - visualization
with col1:
    st.subheader("Interactive Visualization")
    
    # Create a placeholder for the matplotlib figure
    fig_placeholder = st.empty()
    
    # Function to update the visualization
    def update_visualization():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Colors for visualization
        colors = [
            '#ff6b6b', '#48dbfb', '#1dd1a1', '#feca57', '#54a0ff',
            '#5f27cd', '#ff9ff3', '#00d2d3', '#576574', '#222f3e'
        ]
        
        # If we have data points
        if len(st.session_state.data) > 0:
            # If we have clusters
            if len(st.session_state.labels) > 0:
                # Plot points with cluster colors
                for i in range(st.session_state.k):
                    cluster_points = st.session_state.data[st.session_state.labels == i]
                    ax.scatter(
                        cluster_points[:, 0], 
                        cluster_points[:, 1], 
                        c=colors[i % len(colors)],
                        alpha=0.7,
                        s=50
                    )
                
                # Plot centroids
                ax.scatter(
                    st.session_state.centroids[:, 0],
                    st.session_state.centroids[:, 1],
                    c='black',
                    s=100,
                    marker='x'
                )
                
                # Optional: draw lines from points to centroids
                for i in range(len(st.session_state.data)):
                    point = st.session_state.data[i]
                    centroid = st.session_state.centroids[st.session_state.labels[i]]
                    ax.plot(
                        [point[0], centroid[0]],
                        [point[1], centroid[1]],
                        color=colors[st.session_state.labels[i] % len(colors)],
                        alpha=0.2
                    )
            else:
                # Just plot the points without clusters
                ax.scatter(
                    st.session_state.data[:, 0],
                    st.session_state.data[:, 1],
                    c='#999',
                    alpha=0.7,
                    s=50
                )
        
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 400)
        ax.set_title("K-means Clustering Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    # Display the current state
    if len(st.session_state.data) > 0:
        fig = update_visualization()
        fig_placeholder.pyplot(fig)
    else:
        fig_placeholder.info("Generate random data or add points manually to begin")
    
    # Add points manually
    st.write("#### Add Points Manually")
    manual_col1, manual_col2 = st.columns(2)
    
    with manual_col1:
        x_coord = st.number_input("X coordinate", min_value=0, max_value=600, value=300, step=10)
    
    with manual_col2:
        y_coord = st.number_input("Y coordinate", min_value=0, max_value=400, value=200, step=10)
    
    if st.button("Add Point"):
        # Add new point to data
        new_point = np.array([[x_coord, y_coord]])
        if len(st.session_state.data) == 0:
            st.session_state.data = new_point
        else:
            st.session_state.data = np.vstack([st.session_state.data, new_point])
        
        # Reset clustering if we're adding new points
        st.session_state.centroids = np.array([])
        st.session_state.labels = np.array([])
        st.session_state.current_iteration = 0
        st.session_state.inertia_history = []
        st.experimental_rerun()

# Right column - controls
with col2:
    st.subheader("Algorithm Controls")
    
    with st.expander("Parameters", expanded=True):
        # K value (number of clusters)
        st.session_state.k = st.slider(
            "Number of clusters (k)", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.k,
            disabled=st.session_state.is_running
        )
        
        # Max iterations
        st.session_state.max_iterations = st.number_input(
            "Max iterations", 
            min_value=1, 
            max_value=50, 
            value=st.session_state.max_iterations,
            disabled=st.session_state.is_running
        )
        
        # Animation speed control (only shown when running)
        if st.session_state.is_running:
            st.session_state.animation_speed = st.slider(
                "Animation speed (seconds)", 
                min_value=0.1, 
                max_value=2.0, 
                value=st.session_state.animation_speed,
                step=0.1
            )
    
    # Control buttons based on current state
    if st.session_state.is_running:
        if st.button("Stop Algorithm", type="primary"):
            st.session_state.is_running = False
            st.experimental_rerun()
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Initialize", disabled=len(st.session_state.data) == 0):
                initialize_centroids()
                st.experimental_rerun()
        
        with col2:
            step_disabled = (len(st.session_state.centroids) == 0 or 
                           st.session_state.current_iteration >= st.session_state.max_iterations)
            if st.button("Step", disabled=step_disabled):
                run_iteration()
                st.experimental_rerun()
        
        # Auto-run button
        if len(st.session_state.centroids) > 0 and st.session_state.current_iteration < st.session_state.max_iterations:
            if st.button("Auto-run Algorithm", type="primary"):
                st.session_state.is_running = True
                st.experimental_rerun()
    
    # Data generation and clearing
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Random Data", disabled=st.session_state.is_running):
            generate_random_data()
            st.experimental_rerun()
    
    with col2:
        if st.button("Clear All", disabled=st.session_state.is_running):
            clear_data()
            st.experimental_rerun()
    
    # Status information
    st.subheader("Status")
    st.write(f"Data points: **{len(st.session_state.data)}**")
    st.write(f"Iteration: **{st.session_state.current_iteration} / {st.session_state.max_iterations}**")
    
    if len(st.session_state.inertia_history) > 0:
        current_inertia = st.session_state.inertia_history[-1]["inertia"]
        st.write(f"Inertia: **{current_inertia:.2f}** (lower is better)")

# Display learning progress chart if we have history data
if len(st.session_state.inertia_history) > 1:
    st.subheader("Learning Progress")
    
    # Convert history to DataFrame
    history_df = pd.DataFrame(st.session_state.inertia_history)
    
    # Display chart
    st.line_chart(
        history_df.set_index("iteration")["inertia"], 
        use_container_width=True
    )
    
    st.caption(
        "The chart shows how the algorithm improves with each iteration. "
        "Lower inertia means better clustering."
    )

# Educational section
st.subheader("How K-means Clustering Works")
col1, col2 = st.columns(2)

with col1:
    st.write("#### The Algorithm:")
    st.markdown("""
    1. Initialize k random cluster centers (centroids)
    2. Assign each data point to the nearest centroid
    3. Recalculate centroids as the average of all points in the cluster
    4. Repeat steps 2-3 until centroids no longer move significantly
    """)

with col2:
    st.write("#### What You're Seeing:")
    st.markdown("""
    - Points are colored based on which cluster they belong to
    - X markers show the centroids (cluster centers)
    - Lines connect points to their assigned centroid
    - The chart shows how error decreases with each iteration
    - K-means is **unsupervised** - it finds patterns without labels
    """)

st.info("""
#### Real-world Applications:
- Customer segmentation in marketing
- Image compression and color quantization
- Anomaly detection in cybersecurity
- Document clustering for topic modeling
- Genetic clustering in biology
""")

    # Auto-run loop
if st.session_state.is_running:
    # Check if we should continue running
    if st.session_state.current_iteration >= st.session_state.max_iterations:
        st.session_state.is_running = False
        st.experimental_rerun()
    else:
        # Run one iteration
        changed = run_iteration()
        
        # Update the visualization
        fig = update_visualization()
        fig_placeholder.pyplot(fig)
        
        # Stop if convergence or max iterations reached
        if not changed or st.session_state.current_iteration >= st.session_state.max_iterations:
            st.session_state.is_running = False
            st.experimental_rerun()
        else:
            # Sleep for the animation delay
            time.sleep(st.session_state.animation_speed)