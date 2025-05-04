import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

# Set page title and configuration
st.set_page_config(page_title="K-means Clustering Demo", layout="wide")
st.title("K-means Clustering - Machine Learning in Action")

# Make sure to initialize session state variables
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
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame({'X': [], 'Y': []})
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Data Input"

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
    
    # Also update the dataframe
    df = pd.DataFrame(new_data, columns=['X', 'Y'])
    st.session_state.df = df
    
    # Reset clustering
    st.session_state.centroids = np.array([])
    st.session_state.labels = np.array([])
    st.session_state.current_iteration = 0
    st.session_state.inertia_history = []

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
    initial_inertia = kmeans.inertia_
    st.session_state.inertia_history = [{"iteration": 0, "inertia": initial_inertia}]

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

# Function to draw the current state
def draw_visualization():
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
                if len(cluster_points) > 0:
                    ax.scatter(
                        cluster_points[:, 0], 
                        cluster_points[:, 1], 
                        c=colors[i % len(colors)],
                        alpha=0.7,
                        s=50,
                        label=f'Cluster {i+1}'
                    )
            
            # Plot centroids
            ax.scatter(
                st.session_state.centroids[:, 0],
                st.session_state.centroids[:, 1],
                c='black',
                s=100,
                marker='x',
                label='Centroids'
            )
            
            # Add legend
            ax.legend(loc='upper right')
        else:
            # Just plot the points without clusters
            ax.scatter(
                st.session_state.data[:, 0],
                st.session_state.data[:, 1],
                c='#999',
                alpha=0.7,
                s=50
            )
    
    # Set axis limits with padding
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

# Define tab selection function using radio buttons instead
tab_selection = st.radio("Select Mode:", ["Data Input", "Clustering Visualization"], horizontal=True)
st.session_state.active_tab = tab_selection

# Divider
st.markdown("---")

# Display the selected tab
if st.session_state.active_tab == "Data Input":
    st.header("Data Input")
    
    # Option to upload CSV
    st.subheader("Option 1: Upload a CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file with numerical data", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(uploaded_file)
            
            # Display the data
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Check if suitable for clustering
            if df.shape[1] < 2:
                st.error("CSV file must have at least two columns for clustering.")
            else:
                # Allow user to select which columns to use
                st.write("Select two columns to use for clustering:")
                col1 = st.selectbox("X-axis column", df.columns, index=0)
                col2 = st.selectbox("Y-axis column", df.columns, index=min(1, len(df.columns)-1))
                
                # Button to use this data
                if st.button("Use This Data"):
                    # Extract selected columns and convert to numpy array
                    selected_data = df[[col1, col2]].to_numpy()
                    
                    # Update session state
                    st.session_state.data = selected_data
                    
                    # Create a new dataframe with just the selected columns
                    st.session_state.df = df[[col1, col2]].copy()
                    st.session_state.df.columns = ['X', 'Y']
                    
                    # Reset clustering
                    st.session_state.centroids = np.array([])
                    st.session_state.labels = np.array([])
                    st.session_state.current_iteration = 0
                    st.session_state.inertia_history = []
                    
                    st.success("Data loaded successfully!")
                    
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    
    st.markdown("---")
    
    # Option to generate random data
    st.subheader("Option 2: Generate random clustered data")
    
    # Button to generate random data
    if st.button("Generate Random Data"):
        generate_random_data()
        st.success("Random data generated successfully!")
        
        # Show a preview
        st.write("Preview of generated data:")
        st.dataframe(st.session_state.df.head())
    
    st.markdown("---")
    
    # Option to create or edit data in a table
    st.subheader("Option 3: Create or edit data in a table")
    
    # Display the current data as an editable dataframe
    edited_df = st.data_editor(
        st.session_state.df if not st.session_state.df.empty else pd.DataFrame({'X': [], 'Y': []}),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True
    )
    
    # Button to use the edited data
    if st.button("Use Table Data"):
        if not edited_df.empty:
            # Check if there are enough rows
            if len(edited_df) < 2:
                st.error("Please add at least two data points.")
            else:
                # Update session state
                st.session_state.df = edited_df.copy()
                st.session_state.data = edited_df[['X', 'Y']].to_numpy()
                
                # Reset clustering
                st.session_state.centroids = np.array([])
                st.session_state.labels = np.array([])
                st.session_state.current_iteration = 0
                st.session_state.inertia_history = []
                
                st.success("Data from table loaded successfully!")

elif st.session_state.active_tab == "Clustering Visualization":
    # Create two columns for the main content
    viz_col, control_col = st.columns([2, 1])
    
    # Left column - visualization
    with viz_col:
        st.subheader("K-means Clustering Visualization")
        
        # Create a placeholder for the matplotlib figure
        fig_placeholder = st.empty()
        
        # Display the current state
        if len(st.session_state.data) > 0:
            fig = draw_visualization()
            fig_placeholder.pyplot(fig)
        else:
            fig_placeholder.info("No data available. Please go to the Data Input tab to add data.")
    
    # Right column - controls
    with control_col:
        st.subheader("Algorithm Controls")
        
        if len(st.session_state.data) == 0:
            st.warning("Please add data in the Data Input tab first.")
        else:
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
                    st.rerun()
            else:
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    if st.button("Initialize", disabled=len(st.session_state.data) == 0):
                        initialize_centroids()
                        st.rerun()
                
                with btn_col2:
                    step_disabled = (len(st.session_state.centroids) == 0 or 
                                  st.session_state.current_iteration >= st.session_state.max_iterations)
                    if st.button("Step", disabled=step_disabled):
                        run_iteration()
                        st.rerun()
                
                # Auto-run button
                if len(st.session_state.centroids) > 0 and st.session_state.current_iteration < st.session_state.max_iterations:
                    if st.button("Auto-run Algorithm", type="primary"):
                        st.session_state.is_running = True
                        st.rerun()
            
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

# Educational section at the bottom of the page
st.markdown("---")
st.subheader("How K-means Clustering Works")
edu_col1, edu_col2 = st.columns(2)

with edu_col1:
    st.write("#### The Algorithm:")
    st.markdown("""
    1. Initialize k random cluster centers (centroids)
    2. Assign each data point to the nearest centroid
    3. Recalculate centroids as the average of all points in the cluster
    4. Repeat steps 2-3 until centroids no longer move significantly
    """)

with edu_col2:
    st.write("#### What You're Seeing:")
    st.markdown("""
    - Points are coloured based on which cluster they belong to
    - X markers show the centroids (cluster centers)
    - The chart shows how error decreases with each iteration
    - K-means is **unsupervised** - it finds patterns without labels
    """)

st.info("""
#### Real-world Applications:
- Customer segmentation in marketing
- Image compression and colour quantization
- Anomaly detection in cybersecurity
- Document clustering for topic modelling
- Genetic clustering in biology
""")

# Auto-run loop
if st.session_state.is_running:
    # Check if we should continue running
    if st.session_state.current_iteration >= st.session_state.max_iterations:
        st.session_state.is_running = False
        st.rerun()
    else:
        # Run one iteration
        changed = run_iteration()
        
        # Update the visualization
        fig = draw_visualization()
        fig_placeholder.pyplot(fig)
        
        # Stop if convergence or max iterations reached
        if not changed or st.session_state.current_iteration >= st.session_state.max_iterations:
            st.session_state.is_running = False
            st.rerun()
        else:
            # Sleep for the animation delay
            time.sleep(st.session_state.animation_speed)