import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random

# Load dataset
df = pd.read_csv("bookstore_visitors_synthetic.csv")

# Only use numeric features for clustering
X = df[['browsing_time_sec', 'num_books_bought']].values

# Optional: scale features for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K value
k = 2
max_iters = 20  # Limit to 5 iterations for demo

# Step 1: Randomly choose initial centroids
random.seed(42)
initial_centroids_idx = random.sample(range(len(X_scaled)), k)
centroids = X_scaled[initial_centroids_idx]

# Function to assign clusters
def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

# Plotting function
def plot_clusters(X, labels, centroids, iteration):
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=150, marker='X', label='Centroids')
    plt.title(f'Iteration {iteration}: K-Means Clustering')
    plt.xlabel('Browsing Time (scaled)')
    plt.ylabel('Number of Books Bought (scaled)')
    plt.legend()
    plt.grid(True)
    plt.show()

# K-Means loop (manual)
for i in range(max_iters):
    print(f"Iteration {i+1}")
    labels = assign_clusters(X_scaled, centroids)
    plot_clusters(X_scaled, labels, centroids, i+1)

    # Step 3: Recompute centroids
    new_centroids = np.array([X_scaled[labels == j].mean(axis=0) for j in range(k)])

    # Stop if converged
    if np.allclose(new_centroids, centroids):
        print("Converged.")
        break

    centroids = new_centroids
