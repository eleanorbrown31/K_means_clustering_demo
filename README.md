# K_means_clustering_demo
# K-means Clustering - Machine Learning in Action

## Interactive Machine Learning Demo

This Streamlit application demonstrates K-means clustering, an unsupervised machine learning algorithm that groups data points into clusters.

## Features

- **Interactive data generation**: Create random clustered data or add individual points manually
- **Step-by-step execution**: Watch the algorithm work iteration by iteration
- **Visual feedback**: See how points are assigned to clusters with colour-coding
- **Learning metrics**: Track inertia (error) as the algorithm runs
- **Educational content**: Learn how K-means works and its real-world applications

## Running the Demo

### Online

You can access this demo directly through Streamlit Cloud at: [Your Streamlit App URL]

### Locally

1. Clone this repository:
   ```
   git clone https://github.com/YourUsername/kmeans-clustering-demo.git
   cd kmeans-clustering-demo
   ```

2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## How to Use

1. Generate random data or add points manually
2. Set the number of clusters (k) and maximum iterations
3. Click "Initialize" to place initial centroids randomly
4. Either:
   - Click "Step" to run one iteration at a time
   - Click "Auto-run Algorithm" to watch the full process
5. Observe how the points cluster and inertia decreases

## Understanding K-means Clustering

K-means is one of the simplest and most popular unsupervised machine learning algorithms. It works by:

1. Placing k centroids randomly in the data space
2. Assigning each point to its nearest centroid to form clusters
3. Moving each centroid to the average position of all points in its cluster
4. Repeating steps 2-3 until centroids stabilize or the maximum iterations are reached

## Real-world Applications

- Customer segmentation in marketing
- Image compression and colour quantization
- Anomaly detection in cybersecurity
- Document clustering for topic modelling
- Genetic clustering in biology

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Required Python packages

## Requirements

- Python 3.7+
- Dependencies listed in requirements.txt

## License

This project is available under the MIT License.
